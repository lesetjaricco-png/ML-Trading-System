"""Run one CPU-safe MLP directional test on the frozen 42-feature dataset."""

from __future__ import annotations

import ctypes
import json
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from diagnose_baseline import load_train_validation_only, sha256
from run_ablation_25f import (
    BASELINE_DIR,
    BASELINE_ID,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DATASET_PATH,
    METADATA_PATH,
    ensure_output_available,
)


EXPERIMENT_ID = f"{BASELINE_ID}_neural_network_directional_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
DIRECTIONAL_REPORT_PATH = (
    BASELINE_DIR
    / f"{BASELINE_ID}_directional_signal_diagnostic_v1"
    / "diagnostic_report.json"
)
MODEL_PATH = OUTPUT_DIR / "mlp_model.joblib"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
RANDOM_SEED = 42
HIDDEN_LAYERS = (32, 16)
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5
INTERNAL_VALIDATION_FRACTION = 0.10
LEARNING_RATE = 0.001
MIN_DELTA = 1e-5
MIN_AVAILABLE_RAM_MIB = 128
ARRAY_MEMORY_HEADROOM_MULTIPLIER = 8
EXPECTED_PARAMETER_COUNT = 1921
EXPECTED_DATASET_SHA256 = "35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634"
EXPECTED_BASELINE_MODEL_SHA256 = "6e30c834df78c6448a6987b9427e1a7ee0677eacaf15efd04e82a828d207f3b8"
FROZEN_XGB_DIRECTIONAL_AUC = 0.5129618200774144
PROMISING_VALIDATION_AUC = 0.54
MINIMUM_CONTROL_GAIN = 0.02
MINIMUM_INTERNAL_AUC = 0.52
MAXIMUM_AUC_GAP = 0.10


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def memory_observation() -> dict[str, float]:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Unable to read Windows memory status")
    return {
        "total_ram_gib": round(status.total_physical / 2**30, 3),
        "available_ram_gib": round(status.available_physical / 2**30, 3),
        "available_ram_mib": round(status.available_physical / 2**20, 1),
    }


def parameter_count(input_features: int = 42) -> int:
    dimensions = (input_features, *HIDDEN_LAYERS, 1)
    return sum((left + 1) * right for left, right in zip(dimensions, dimensions[1:]))


def validate_manifest(features: list[str]) -> None:
    invalid = [
        feature
        for feature in features
        if feature == "target"
        or feature.startswith("future_")
        or feature.endswith("_outcome")
    ]
    if len(features) != 42 or len(set(features)) != 42 or invalid:
        raise RuntimeError(f"Invalid frozen feature manifest: {invalid}")
    if parameter_count(len(features)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("MLP parameter count differs from the predeclared architecture")


def resource_preflight(train_rows: int, validation_rows: int) -> dict[str, Any]:
    memory = memory_observation()
    estimated_arrays_mib = (train_rows + validation_rows) * 42 * 4 / 2**20
    required_available_mib = max(
        MIN_AVAILABLE_RAM_MIB,
        ARRAY_MEMORY_HEADROOM_MULTIPLIER * estimated_arrays_mib,
    )
    if memory["available_ram_mib"] < required_available_mib:
        raise MemoryError(
            f"Only {memory['available_ram_mib']:.1f} MiB RAM is available; "
            f"at least {required_available_mib:.1f} MiB is required"
        )
    if (
        HIDDEN_LAYERS != (32, 16)
        or BATCH_SIZE != 256
        or MAX_EPOCHS != 50
        or PATIENCE != 5
        or parameter_count() != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("Resource-safe MLP manifest changed")
    return {
        **memory,
        "dataset_shape_exposed": [train_rows + validation_rows, 43],
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "feature_count": 42,
        "estimated_parameter_count": parameter_count(),
        "estimated_float32_feature_arrays_mib": round(estimated_arrays_mib, 2),
        "required_available_ram_mib": round(required_available_mib, 2),
        "array_memory_headroom_multiplier": ARRAY_MEMORY_HEADROOM_MULTIPLIER,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
    }


def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=BATCH_SIZE,
        learning_rate="constant",
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        shuffle=False,
        random_state=RANDOM_SEED,
        early_stopping=False,
        warm_start=False,
    )


def chronological_internal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_position = int(len(frame) * (1 - INTERNAL_VALIDATION_FRACTION))
    fit = frame.iloc[:split_position]
    holdout = frame.iloc[split_position:]
    if fit.empty or holdout.empty or fit.index[-1] >= holdout.index[0]:
        raise RuntimeError("Internal TRAIN holdout is not strictly chronological")
    if set(fit["target"].unique()) != {0, 1} or set(holdout["target"].unique()) != {0, 1}:
        raise RuntimeError("Internal TRAIN split must contain SELL and BUY")
    return fit, holdout


def binary_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(np.int8)
    counts = np.bincount(prediction, minlength=2)
    return {
        "roc_auc": float(roc_auc_score(actual, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, average="macro", zero_division=0)),
        "sell_recall": float(recall_score(actual, prediction, pos_label=0, zero_division=0)),
        "buy_recall": float(recall_score(actual, prediction, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, prediction, labels=[0, 1]).tolist(),
        "prediction_distribution": {
            "SELL": {"count": int(counts[0]), "percentage": float(100 * counts[0] / len(prediction))},
            "BUY": {"count": int(counts[1]), "percentage": float(100 * counts[1] / len(prediction))},
        },
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std(ddof=0)),
        "predictions_near_0_5_pct": float(100 * np.mean((probability >= 0.45) & (probability <= 0.55))),
    }


def classify_result(
    train_metrics: dict[str, Any],
    internal_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    gap = train_metrics["roc_auc"] - validation_metrics["roc_auc"]
    flags = {
        "severe_overfitting": gap > MAXIMUM_AUC_GAP,
        "cross_period_instability": abs(
            internal_metrics["roc_auc"] - validation_metrics["roc_auc"]
        ) > 0.10,
        "near_constant_predictions": validation_metrics["probability_std"] < 0.02,
        "extreme_class_bias": max(
            item["percentage"]
            for item in validation_metrics["prediction_distribution"].values()
        ) > 90,
    }
    signal = (
        validation_metrics["roc_auc"] >= PROMISING_VALIDATION_AUC
        and validation_metrics["roc_auc"] - FROZEN_XGB_DIRECTIONAL_AUC >= MINIMUM_CONTROL_GAIN
        and internal_metrics["roc_auc"] >= MINIMUM_INTERNAL_AUC
        and gap <= MAXIMUM_AUC_GAP
        and not any(flags.values())
    )
    return {
        "verdict": "NEURAL_DIRECTIONAL_SIGNAL" if signal else "NEURAL_NO_DIRECTIONAL_SIGNAL",
        "train_to_external_validation_auc_gap": gap,
        "flags": flags,
        "recommendation": (
            "A separately approved sequential-model experiment may be considered."
            if signal
            else "Retire the neural-network branch; do not proceed automatically to LSTM."
        ),
    }


def protected_hashes(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    return {
        str(path).replace("\\", "/"): sha256(path)
        for root in (Path("data"), BASELINE_DIR)
        for path in sorted(root.rglob("*"))
        if path.is_file() and output_dir not in path.parents
    }


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    np.random.seed(RANDOM_SEED)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    directional_report = json.loads(DIRECTIONAL_REPORT_PATH.read_text(encoding="utf-8"))
    features = list(baseline_report["features"]["names_in_order"])
    validate_manifest(features)
    if features != list(metadata["feature_names"]):
        raise RuntimeError("Feature order differs from frozen dataset metadata")
    if sha256(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Frozen dataset hash mismatch")
    if sha256(BASELINE_MODEL_PATH) != EXPECTED_BASELINE_MODEL_SHA256:
        raise RuntimeError("Frozen XGBoost model hash mismatch")
    recorded_control = directional_report["saved_model_validation"]["reference_42f"]
    if recorded_control["buy_vs_sell_on_actual_directional"]["roc_auc"] != FROZEN_XGB_DIRECTIONAL_AUC:
        raise RuntimeError("Frozen XGBoost directional control changed")
    prior_before = protected_hashes(output_dir)
    train, validation, access = load_train_validation_only(
        DATASET_PATH, metadata, [*features, "target"]
    )
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("Sealed loader exposed TEST")
    if train.index[-1] >= validation.index[0]:
        raise RuntimeError("External VALIDATION does not follow TRAIN")
    resources = resource_preflight(len(train), len(validation))
    scaler = StandardScaler(copy=True)
    scaler.fit(train[features].to_numpy(dtype=np.float32, copy=False))
    if int(scaler.n_samples_seen_) != len(train):
        raise RuntimeError("Scaler did not see exactly the complete TRAIN partition")
    directional_train = train.loc[train["target"].isin([0, 1])]
    directional_validation = validation.loc[validation["target"].isin([0, 1])]
    fit_frame, internal_frame = chronological_internal_split(directional_train)
    x_fit = scaler.transform(fit_frame[features].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    y_fit = fit_frame["target"].to_numpy(dtype=np.int8, copy=True)
    x_internal = scaler.transform(internal_frame[features].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    y_internal = internal_frame["target"].to_numpy(dtype=np.int8, copy=True)
    x_validation = scaler.transform(directional_validation[features].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    y_validation = directional_validation["target"].to_numpy(dtype=np.int8, copy=True)
    del train, validation, directional_train
    model = build_model()
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(x_fit, y_fit, classes=np.asarray([0, 1], dtype=np.int8))
        fit_probability = model.predict_proba(x_fit)[:, 1]
        internal_probability = model.predict_proba(x_internal)[:, 1]
        fit_loss = float(log_loss(y_fit, fit_probability, labels=[0, 1]))
        internal_loss = float(log_loss(y_internal, internal_probability, labels=[0, 1]))
        row = {
            "epoch": epoch,
            "train_loss": fit_loss,
            "internal_validation_loss": internal_loss,
            "train_auc": float(roc_auc_score(y_fit, fit_probability)),
            "internal_validation_auc": float(roc_auc_score(y_internal, internal_probability)),
        }
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError("Non-finite MLP training history")
        history.append(row)
        if internal_loss < best_loss - MIN_DELTA:
            best_loss = internal_loss
            best_epoch = epoch
            best_state = (deepcopy(model.coefs_), deepcopy(model.intercepts_))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("MLP training produced no valid epoch")
    model.coefs_, model.intercepts_ = best_state
    train_probability = model.predict_proba(x_fit)[:, 1]
    internal_probability = model.predict_proba(x_internal)[:, 1]
    validation_probability = model.predict_proba(x_validation)[:, 1]
    train_metrics = binary_metrics(y_fit, train_probability)
    internal_metrics = binary_metrics(y_internal, internal_probability)
    validation_metrics = binary_metrics(y_validation, validation_probability)
    decision = classify_result(train_metrics, internal_metrics, validation_metrics)
    output_dir.mkdir(parents=True)
    joblib.dump({"model": model, "scaler": scaler, "features": features}, MODEL_PATH if output_dir == OUTPUT_DIR else output_dir / MODEL_PATH.name)
    scaler_file = SCALER_PATH if output_dir == OUTPUT_DIR else output_dir / SCALER_PATH.name
    joblib.dump(scaler, scaler_file)
    model_file = MODEL_PATH if output_dir == OUTPUT_DIR else output_dir / MODEL_PATH.name
    reloaded = joblib.load(model_file)
    reload_probability = reloaded["model"].predict_proba(
        reloaded["scaler"].transform(
            directional_validation[features].to_numpy(dtype=np.float32, copy=False)
        ).astype(np.float32, copy=False)
    )[:, 1]
    maximum_reload_difference = float(np.max(np.abs(validation_probability - reload_probability)))
    reload_identical = bool(np.array_equal(validation_probability, reload_probability))
    if maximum_reload_difference > 1e-12:
        raise RuntimeError("Reloaded MLP predictions materially differ")
    prior_after = {path: sha256(Path(path)) for path in prior_before}
    if prior_before != prior_after:
        raise RuntimeError("A protected artifact changed")
    scaler_metadata = {
        "type": "StandardScaler",
        "fit_rows": int(scaler.n_samples_seen_),
        "feature_order": features,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "artifact": scaler_file.name,
        "artifact_sha256": sha256(scaler_file),
    }
    (output_dir / "scaler_metadata.json").write_text(
        json.dumps(scaler_metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    predictions = pd.DataFrame(
        {
            "actual": y_validation,
            "buy_probability": validation_probability.astype(np.float32),
            "prediction": (validation_probability >= 0.5).astype(np.int8),
        },
        index=directional_validation.index,
    )
    predictions.to_parquet(output_dir / "validation_predictions.parquet")
    metrics = {
        "train": train_metrics,
        "internal_train_validation": internal_metrics,
        "external_validation": validation_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_question": "Can one small MLP recover BUY-vs-SELL direction from the frozen 42-feature representation?",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "execution": "CPU-only; no TensorFlow, CUDA, GPU, or external data",
            "resources_before_training": resources,
        },
        "predeclared_design": {
            "architecture": "42 -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, logistic)",
            "dropout": "unavailable in sklearn MLPClassifier; omitted as predeclared",
            "parameter_count": parameter_count(),
            "solver": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "internal_validation_fraction": INTERNAL_VALIDATION_FRACTION,
            "random_seed": RANDOM_SEED,
            "shuffle": False,
            "class_weighting": False,
            "probability_calibration": False,
            "classification_threshold": 0.5,
            "architecture_search": False,
            "hyperparameter_search": False,
            "threshold_optimization": False,
            "cv_refits": 0,
            "decision_thresholds": {
                "minimum_validation_auc": PROMISING_VALIDATION_AUC,
                "minimum_gain_over_frozen_xgb": MINIMUM_CONTROL_GAIN,
                "minimum_internal_auc": MINIMUM_INTERNAL_AUC,
                "maximum_train_validation_auc_gap": MAXIMUM_AUC_GAP,
            },
        },
        "data": {
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "features": features,
            "feature_count": len(features),
            "target": "frozen v0.3 endpoint target restricted to SELL=0 and BUY=1",
            "train_rows_exposed": int(metadata["splits"]["train"]["row_count"]),
            "validation_rows_exposed": int(metadata["splits"]["validation"]["row_count"]),
            "directional_fit_rows": len(fit_frame),
            "directional_internal_validation_rows": len(internal_frame),
            "directional_external_validation_rows": len(directional_validation),
            "internal_fit_last_timestamp": pd.Timestamp(fit_frame.index[-1]).isoformat(),
            "internal_validation_first_timestamp": pd.Timestamp(internal_frame.index[0]).isoformat(),
            "external_validation_first_timestamp": pd.Timestamp(directional_validation.index[0]).isoformat(),
            "data_access": access,
        },
        "training": {
            "actual_epochs": len(history),
            "best_epoch": best_epoch,
            "best_internal_validation_loss": best_loss,
            "final_observed_epoch": history[-1],
            "best_model_train_loss": float(log_loss(y_fit, train_probability, labels=[0, 1])),
            "best_model_internal_validation_loss": float(log_loss(y_internal, internal_probability, labels=[0, 1])),
        },
        "metrics": metrics,
        "comparison": {
            "chance_directional_auc": 0.5,
            "frozen_xgboost_42f": {
                "validation_directional_auc": FROZEN_XGB_DIRECTIONAL_AUC,
                "chronological_cv_directional_auc": None,
                "directional_balanced_accuracy": recorded_control["buy_vs_sell_on_actual_directional"]["balanced_accuracy"],
                "directional_macro_f1": None,
                "source": str(DIRECTIONAL_REPORT_PATH),
            },
            "small_mlp_42f": {
                "validation_directional_auc": validation_metrics["roc_auc"],
                "chronological_cv_directional_auc": None,
                "chronological_internal_holdout_auc": internal_metrics["roc_auc"],
                "directional_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "directional_macro_f1": validation_metrics["macro_f1"],
            },
        },
        "decision": decision,
        "integrity": {
            "protected_artifacts_checked": len(prior_before),
            "protected_artifacts_unchanged": True,
            "dataset_hash_verified": True,
            "frozen_xgboost_identity_verified": True,
            "exact_feature_order_verified": True,
            "scaler_train_rows": int(scaler.n_samples_seen_),
            "scaler_fit_on_train_only": True,
            "external_validation_used_for_training": False,
            "test_rows_exposed": 0,
            "test_features_read": False,
            "test_labels_read": False,
            "target_absent_from_features": True,
            "future_columns_absent": True,
            "chronology_verified": True,
            "model_instances_trained": 1,
            "reload_predictions_identical": reload_identical,
            "maximum_reload_probability_difference": maximum_reload_difference,
            "deterministic_seed_configured": True,
            "model_sha256": sha256(model_file),
        },
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        "## Verdict",
        f"**{decision['verdict']}**",
        "",
        "## Directional Comparison",
        "| Model | Validation AUC | Chronological evidence | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Frozen XGBoost 42f | {FROZEN_XGB_DIRECTIONAL_AUC:.4f} | unavailable | {recorded_control['buy_vs_sell_on_actual_directional']['balanced_accuracy']:.4f} | unavailable |",
        f"| Small MLP 42f | {validation_metrics['roc_auc']:.4f} | internal holdout {internal_metrics['roc_auc']:.4f} | {validation_metrics['balanced_accuracy']:.4f} | {validation_metrics['macro_f1']:.4f} |",
        "",
        "## Training",
        f"- Epochs / best epoch: {len(history)} / {best_epoch}",
        f"- TRAIN AUC: {train_metrics['roc_auc']:.4f}",
        f"- Internal TRAIN-validation AUC: {internal_metrics['roc_auc']:.4f}",
        f"- External VALIDATION AUC: {validation_metrics['roc_auc']:.4f}",
        f"- TRAIN to external gap: {decision['train_to_external_validation_auc_gap']:.4f}",
        "",
        "## Integrity",
        "- TEST rows/features/labels exposed: 0 / false / false",
        f"- Scaler TRAIN rows: {int(scaler.n_samples_seen_):,}",
        f"- Reload predictions identical: {reload_identical}",
        f"- Protected artifacts unchanged: true ({len(prior_before)} checked)",
        "",
        decision["recommendation"],
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment()["decision"], indent=2))