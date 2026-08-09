"""Train the first leakage-safe V0.3 XGBoost baseline from processed data."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.model import XGBoostModel
from src.utils import load_config


BASELINE_ID = "v0.3_forward_atr_xgb_baseline_v1"
DATASET_PATH = Path(
    "data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.parquet"
)
METADATA_PATH = DATASET_PATH.with_suffix(".metadata.json")
CONFIG_PATH = Path("config/config_v03.yaml")
OUTPUT_DIR = Path("models/baselines")
CLASS_NAMES = {0: "SELL", 1: "BUY", 2: "NO_TRADE"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chronological_partitions(
    dataset: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the recorded TRAIN and VALIDATION partitions; leave TEST unused."""
    split_metadata = metadata["splits"]
    train_count = int(split_metadata["train"]["row_count"])
    validation_count = int(split_metadata["validation"]["row_count"])
    test_count = int(split_metadata["test"]["row_count"])
    if train_count + validation_count + test_count != len(dataset):
        raise ValueError("Recorded split counts do not match the processed dataset")
    if not dataset.index.is_monotonic_increasing or not dataset.index.is_unique:
        raise ValueError("Processed dataset timestamps must be unique and chronological")

    validation_end = train_count + validation_count
    train = dataset.iloc[:train_count]
    validation = dataset.iloc[train_count:validation_end]
    expected_boundaries = {
        "train_first": split_metadata["train"]["first_timestamp"],
        "train_last": split_metadata["train"]["last_timestamp"],
        "validation_first": split_metadata["validation"]["first_timestamp"],
        "validation_last": split_metadata["validation"]["last_timestamp"],
    }
    actual_boundaries = {
        "train_first": pd.Timestamp(train.index[0]).isoformat(),
        "train_last": pd.Timestamp(train.index[-1]).isoformat(),
        "validation_first": pd.Timestamp(validation.index[0]).isoformat(),
        "validation_last": pd.Timestamp(validation.index[-1]).isoformat(),
    }
    if actual_boundaries != expected_boundaries:
        raise ValueError("Recorded split timestamps do not match the processed dataset")
    return train, validation


def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return classification and directional metrics for labels 0, 1, and 2."""
    labels = list(CLASS_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    class_metrics = {
        CLASS_NAMES[label]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    prediction_counts = pd.Series(y_pred).value_counts().reindex(labels, fill_value=0)
    directional_mask = y_true != 2
    directional_accuracy = accuracy_score(
        y_true[directional_mask], y_pred[directional_mask]
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": class_metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[label] for label in labels],
        "prediction_distribution": {
            CLASS_NAMES[label]: {
                "count": int(prediction_counts[label]),
                "percentage": float(100 * prediction_counts[label] / len(y_pred)),
            }
            for label in labels
        },
        "directional": {
            "accuracy_on_actual_buy_sell_rows": float(directional_accuracy),
            "actual_directional_row_count": int(directional_mask.sum()),
            "predicted_buy_sell_percentage": float(100 * np.mean(y_pred != 2)),
            "predicted_no_trade_percentage": float(100 * np.mean(y_pred == 2)),
            "buy_precision": class_metrics["BUY"]["precision"],
            "buy_recall": class_metrics["BUY"]["recall"],
            "sell_precision": class_metrics["SELL"]["precision"],
            "sell_recall": class_metrics["SELL"]["recall"],
        },
    }


def _model_parameters(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "random_state",
        "eval_metric",
    )
    return {key: config["model"][key] for key in keys}


def _checks_pass(checks: dict[str, bool | int]) -> bool:
    return (
        checks["test_evaluated"] is False
        and all(
            value is True
            for name, value in checks.items()
            if name not in {"test_evaluated", "scaler_fit_row_count"}
        )
    )


def _render_report(report: dict[str, Any]) -> str:
    train = report["evaluation"]["train"]
    validation = report["evaluation"]["validation"]
    majority = report["baselines"]["majority_class"]
    gap = report["train_validation_gap"]
    cv_folds = report["chronological_cv"]["folds"]
    importance = report["feature_importance"][:15]
    lines = [
        f"# {BASELINE_ID}",
        "",
        "## Scope",
        f"- Dataset: `{report['dataset']['identifier']}`",
        f"- Train rows: {report['splits']['train']['row_count']:,}",
        f"- Validation rows: {report['splits']['validation']['row_count']:,}",
        "- Test rows were not evaluated.",
        f"- Training duration: {report['training']['duration_seconds']:.2f} seconds",
        "",
        "## Metrics",
        "| Split | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
        f"| Train | {train['accuracy']:.4f} | {train['balanced_accuracy']:.4f} | {train['macro_f1']:.4f} |",
        f"| Validation | {validation['accuracy']:.4f} | {validation['balanced_accuracy']:.4f} | {validation['macro_f1']:.4f} |",
        f"| Majority ({majority['class_name']}) | {majority['validation']['accuracy']:.4f} | {majority['validation']['balanced_accuracy']:.4f} | {majority['validation']['macro_f1']:.4f} |",
        "",
        "## Validation Per Class",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name, metrics in validation["per_class"].items():
        lines.append(
            f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['support']:,} |"
        )
    lines.extend(
        [
            "",
            "## Prediction Distribution",
            "| Split | SELL | BUY | NO_TRADE |",
            "|---|---:|---:|---:|",
            f"| Train | {train['prediction_distribution']['SELL']['percentage']:.2f}% | "
            f"{train['prediction_distribution']['BUY']['percentage']:.2f}% | "
            f"{train['prediction_distribution']['NO_TRADE']['percentage']:.2f}% |",
            f"| Validation | {validation['prediction_distribution']['SELL']['percentage']:.2f}% | "
            f"{validation['prediction_distribution']['BUY']['percentage']:.2f}% | "
            f"{validation['prediction_distribution']['NO_TRADE']['percentage']:.2f}% |",
            "",
            "## Directional Validation",
            f"- Accuracy on actual BUY/SELL rows: {validation['directional']['accuracy_on_actual_buy_sell_rows']:.4f}",
            f"- Predicted BUY/SELL: {validation['directional']['predicted_buy_sell_percentage']:.2f}%",
            f"- Predicted NO_TRADE: {validation['directional']['predicted_no_trade_percentage']:.2f}%",
            f"- BUY precision/recall: {validation['directional']['buy_precision']:.4f} / {validation['directional']['buy_recall']:.4f}",
            f"- SELL precision/recall: {validation['directional']['sell_precision']:.4f} / {validation['directional']['sell_recall']:.4f}",
            "",
            "## Generalization Gap",
            f"- Accuracy: {gap['accuracy']:.4f}",
            f"- Balanced accuracy: {gap['balanced_accuracy']:.4f}",
            f"- Macro F1: {gap['macro_f1']:.4f}",
            "",
            "## TRAIN-only Chronological CV",
            f"- Five fold accuracy range: {min(fold['accuracy'] for fold in cv_folds):.4f} to {max(fold['accuracy'] for fold in cv_folds):.4f}",
            f"- Mean accuracy: {np.mean([fold['accuracy'] for fold in cv_folds]):.4f}",
            f"- Mean weighted F1: {np.mean([fold['f1'] for fold in cv_folds]):.4f}",
            "",
            "## Validation Confusion Matrix",
            "Rows are actual labels; columns are predicted labels in SELL, BUY, NO_TRADE order.",
            "",
            "```text",
            *[str(row) for row in validation["confusion_matrix"]],
            "```",
            "",
            "## Top Native Feature Importances",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ]
    )
    for rank, item in enumerate(importance, 1):
        lines.append(f"| {rank} | {item['feature']} | {item['importance']:.6f} |")
    lines.extend(
        [
            "",
            "## Policy",
            "- Validation was used only for post-fit evaluation.",
            "- Chronological cross-validation used TRAIN only with fold-local scaling.",
            "- The final TEST partition was not evaluated or used for model selection.",
            "- No feature selection, class rebalancing, or hyperparameter optimization was performed.",
            "",
            "## Assessment",
            f"- {report['assessment']['learnability']}",
            f"- Next stage: {report['assessment']['recommendation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_baseline() -> dict[str, Any]:
    """Train, evaluate, cross-validate, persist, and reload the baseline."""
    output_paths = {
        "model": OUTPUT_DIR / f"{BASELINE_ID}.joblib",
        "evaluation": OUTPUT_DIR / f"{BASELINE_ID}_evaluation.json",
        "report": OUTPUT_DIR / f"{BASELINE_ID}_report.md",
        "feature_importance": OUTPUT_DIR / f"{BASELINE_ID}_feature_importance.csv",
        "validation_confusion_matrix": OUTPUT_DIR
        / f"{BASELINE_ID}_validation_confusion_matrix.csv",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Baseline artifacts already exist: {existing}")

    dataset_hash_before = _sha256(DATASET_PATH)
    metadata_hash_before = _sha256(METADATA_PATH)
    with METADATA_PATH.open(encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)
    config = load_config(str(CONFIG_PATH))
    dataset = pd.read_parquet(DATASET_PATH)
    feature_names = list(metadata["feature_names"])
    if metadata["fallback_used"] or metadata["raw_provenance"]["fallback_used"]:
        raise ValueError("Baseline training requires verified non-fallback MT5 data")
    if metadata["feature_count"] != len(feature_names):
        raise ValueError("Feature metadata is inconsistent")
    if not np.isfinite(dataset[feature_names].to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values")

    train, validation = chronological_partitions(dataset, metadata)
    model_parameters = _model_parameters(config)
    model = XGBoostModel(
        **model_parameters,
        signal_mode=config["experiment"]["signal_mode"],
        buy_threshold=config["signal"]["buy_threshold"],
        sell_threshold=config["signal"]["sell_threshold"],
        models_dir=str(OUTPUT_DIR),
    )

    training_started = time.perf_counter()
    model.fit_training_data(train, feature_names)
    training_duration = time.perf_counter() - training_started

    X_train = train[feature_names].to_numpy()
    X_validation = validation[feature_names].to_numpy()
    train_predictions = model.model.predict(model.scaler.transform(X_train))
    validation_predictions = model.model.predict(model.scaler.transform(X_validation))
    train_metrics = evaluation_metrics(train["target"].to_numpy(), train_predictions)
    validation_metrics = evaluation_metrics(
        validation["target"].to_numpy(), validation_predictions
    )

    majority_class = int(train["target"].mode().iloc[0])
    majority_train = np.full(len(train), majority_class)
    majority_validation = np.full(len(validation), majority_class)
    majority_metrics = {
        "class_label": majority_class,
        "class_name": CLASS_NAMES[majority_class],
        "train": evaluation_metrics(train["target"].to_numpy(), majority_train),
        "validation": evaluation_metrics(
            validation["target"].to_numpy(), majority_validation
        ),
    }

    cv_model = XGBoostModel(
        **model_parameters,
        signal_mode=config["experiment"]["signal_mode"],
        models_dir=str(OUTPUT_DIR),
    )
    cv_started = time.perf_counter()
    cv_results = cv_model.cross_validate(train, feature_names, n_splits=5)
    cv_duration = time.perf_counter() - cv_started

    feature_importance = model.feature_importance()
    report: dict[str, Any] = {
        "baseline_id": BASELINE_ID,
        "training": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": training_duration,
            "random_seed": model_parameters["random_state"],
            "preprocessing": (
                "StandardScaler fit on the complete recorded TRAIN partition only; "
                "VALIDATION transformed with TRAIN-fitted parameters; TEST untouched"
            ),
            "early_stopping": "disabled; validation was not supplied to model.fit",
        },
        "model_parameters": {
            **model_parameters,
            "configured_eval_metric": model_parameters["eval_metric"],
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": 3,
            "n_jobs": -1,
        },
        "dataset": {
            "identifier": DATASET_PATH.stem,
            "path": str(DATASET_PATH),
            "metadata_path": str(METADATA_PATH),
            "sha256": dataset_hash_before,
            "metadata_sha256": metadata_hash_before,
            "feature_generation_version": metadata["feature_generation_version"],
            "target_generation_version": metadata["target_generation_version"],
            "source": metadata["source"],
            "fallback_used": metadata["fallback_used"],
        },
        "features": {"count": len(feature_names), "names_in_order": feature_names},
        "splits": metadata["splits"],
        "test_policy": "not evaluated; not used for fitting, preprocessing, CV, or selection",
        "evaluation": {"train": train_metrics, "validation": validation_metrics},
        "baselines": {"majority_class": majority_metrics},
        "train_validation_gap": {
            "accuracy": train_metrics["accuracy"] - validation_metrics["accuracy"],
            "balanced_accuracy": train_metrics["balanced_accuracy"]
            - validation_metrics["balanced_accuracy"],
            "macro_f1": train_metrics["macro_f1"] - validation_metrics["macro_f1"],
        },
        "chronological_cv": {
            "scope": "TRAIN partition only",
            "n_splits": 5,
            "preprocessing": "fresh StandardScaler fit independently on each fold's training rows",
            "duration_seconds": cv_duration,
            "folds": cv_results,
        },
        "assessment": {
            "learnability": (
                "Weak evidence of class discrimination beyond the majority baseline on "
                "balanced accuracy and macro F1, but poor temporal generalization, lower "
                "overall validation accuracy, and near-absent BUY recall prevent calling "
                "the current baseline robustly learnable."
            ),
            "recommendation": (
                "Diagnose temporal drift and session/price-level dependence using TRAIN and "
                "VALIDATION only before any controlled feature or hyperparameter experiment; "
                "keep TEST sealed."
            ),
        },
        "feature_importance": feature_importance.to_dict(orient="records"),
        "checks": {},
        "artifacts": {key: str(path) for key, path in output_paths.items()},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(model.save(BASELINE_ID))
    reloaded = XGBoostModel(models_dir=str(OUTPUT_DIR))
    reloaded.load(BASELINE_ID)
    reloaded_predictions = reloaded.model.predict(
        reloaded.scaler.transform(X_validation)
    )
    report["checks"] = {
        "artifact_reload_predictions_identical": bool(
            np.array_equal(validation_predictions, reloaded_predictions)
        ),
        "repeated_predictions_identical": bool(
            np.array_equal(
                validation_predictions,
                model.model.predict(model.scaler.transform(X_validation)),
            )
        ),
        "feature_order_preserved_after_reload": reloaded.feature_columns
        == feature_names,
        "scaler_fit_row_count": int(model.scaler.n_samples_seen_),
        "scaler_fit_matches_train_row_count": int(model.scaler.n_samples_seen_)
        == len(train),
        "test_evaluated": False,
        "dataset_hash_unchanged": _sha256(DATASET_PATH) == dataset_hash_before,
        "metadata_hash_unchanged": _sha256(METADATA_PATH) == metadata_hash_before,
    }
    if not _checks_pass(report["checks"]):
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Baseline validation checks failed: {report['checks']}")

    feature_importance.to_csv(output_paths["feature_importance"], index=False)
    pd.DataFrame(
        validation_metrics["confusion_matrix"],
        index=validation_metrics["confusion_matrix_labels"],
        columns=validation_metrics["confusion_matrix_labels"],
    ).to_csv(output_paths["validation_confusion_matrix"])
    output_paths["evaluation"].write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    output_paths["report"].write_text(_render_report(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_baseline()
    print(json.dumps(result, indent=2, allow_nan=False))