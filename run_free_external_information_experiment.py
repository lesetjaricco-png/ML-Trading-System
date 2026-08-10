"""Test independently whether acquired free information improves direction."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import time
from datetime import datetime, time as wall_time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from acquire_free_external_information import (
    BASELINE_DIR,
    DATASET_VERSION,
    EXPERIMENT_OUTPUT,
    OUTPUT_ROOT as INFORMATION_ROOT,
)
from diagnose_baseline import DATASET_PATH, METADATA_PATH, load_train_validation_only, sha256
from run_ablation_25f import BASELINE_ID, BASELINE_MODEL_PATH, BASELINE_REPORT_PATH, model_parameters
from src.external_information import validate_feature_names
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_free_external_information_v1"
OUTPUT_DIR = EXPERIMENT_OUTPUT
MANIFEST_PATH = INFORMATION_ROOT / "manifest.json"
PREVIOUS_BEST_AUC = 0.51636723758793
CHANCE_AUC = 0.5
MINIMUM_VALIDATION_GAIN = 0.01
MINIMUM_CV_GAIN = 0.005
MINIMUM_VALIDATION_AUC = 0.525
MINIMUM_POSITIVE_FOLDS = 3
PERMUTATIONS = 199
RANDOM_SEED = 42
NEW_YORK = ZoneInfo("America/New_York")


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


def memory_preflight(rows: int, maximum_features: int) -> dict[str, float]:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Unable to read Windows memory status")
    available_mib = status.available_physical / 2**20
    matrix_mib = rows * maximum_features * 8 / 2**20
    required_mib = max(256.0, matrix_mib * 5)
    if available_mib < required_mib:
        raise MemoryError(
            f"Only {available_mib:.1f} MiB RAM available; experiment requires {required_mib:.1f} MiB"
        )
    return {
        "total_ram_gib": round(status.total_physical / 2**30, 3),
        "available_ram_mib": round(available_mib, 1),
        "estimated_largest_matrix_mib": round(matrix_mib, 2),
        "required_available_ram_mib": round(required_mib, 2),
    }


def protected_hashes() -> dict[str, str]:
    paths = [DATASET_PATH, METADATA_PATH, BASELINE_MODEL_PATH, BASELINE_REPORT_PATH, MANIFEST_PATH]
    paths.extend(
        path
        for root in (INFORMATION_ROOT, BASELINE_DIR)
        for path in root.rglob("*")
        if path.is_file() and OUTPUT_DIR not in path.parents
    )
    return {
        str(path).replace("\\", "/"): sha256(path)
        for path in sorted(set(paths), key=lambda item: str(item))
    }


def _asof_wide(
    prediction_times: pd.DatetimeIndex,
    source: pd.DataFrame,
    maximum_age: pd.Timedelta,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if prediction_times.tz is None or str(prediction_times.tz) != "UTC":
        raise RuntimeError("Prediction timestamps must be timezone-aware UTC")
    if source.index.tz is None or str(source.index.tz) != "UTC":
        raise RuntimeError("Source timestamps must be timezone-aware UTC")
    source = source.sort_index().copy()
    source["maximum_available_time_utc"] = source.index
    left = pd.DataFrame({"prediction_time_utc": prediction_times})
    merged = pd.merge_asof(
        left,
        source.reset_index(drop=True),
        left_on="prediction_time_utc",
        right_on="maximum_available_time_utc",
        direction="backward",
        tolerance=maximum_age,
    )
    valid = merged["maximum_available_time_utc"].notna()
    if (merged.loc[valid, "maximum_available_time_utc"] > merged.loc[valid, "prediction_time_utc"]).any():
        raise RuntimeError("Future source observation detected")
    ages = merged.loc[valid, "prediction_time_utc"] - merged.loc[valid, "maximum_available_time_utc"]
    features = merged.drop(columns=["prediction_time_utc", "maximum_available_time_utc"])
    features.index = prediction_times
    return features, {
        "rows": len(features),
        "matched_rows": int(valid.sum()),
        "future_observations": 0,
        "maximum_age_seconds": None if ages.empty else float(ages.dt.total_seconds().max()),
        "maximum_source_time_utc": None if not valid.any() else merged.loc[valid, "maximum_available_time_utc"].max().isoformat(),
    }


def _load_normalized(path: str) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=[
            "instrument_or_event_id",
            "field",
            "value",
            "source_time_utc",
            "available_time_utc",
            "revision_id",
            "quality_flags",
            "raw_batch_sha256",
        ],
    )


def build_rates_features(
    prediction_times: pd.DatetimeIndex, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = _load_normalized(manifest["sources"]["treasury"]["data_path"])
    rows = rows[rows["field"] == "yield_percent"]
    daily = rows.pivot(index="available_time_utc", columns="instrument_or_event_id", values="value").sort_index()
    features = pd.DataFrame(index=daily.index)
    for maturity in ("2Y", "5Y", "10Y", "30Y"):
        features[f"rates_{maturity.lower()}_level"] = daily[maturity]
        features[f"rates_{maturity.lower()}_daily_change"] = daily[maturity].diff()
    features["rates_2s10s"] = daily["2Y"] - daily["10Y"]
    features["rates_5s30s"] = daily["5Y"] - daily["30Y"]
    features["rates_10y_change_volatility_20d"] = daily["10Y"].diff().rolling(20, min_periods=20).std()
    return _asof_wide(prediction_times, features, pd.Timedelta(days=7))


def build_volatility_features(
    prediction_times: pd.DatetimeIndex, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = _load_normalized(manifest["sources"]["volatility"]["data_path"])
    daily = rows.pivot(index="available_time_utc", columns="instrument_or_event_id", values="value").sort_index()
    features = pd.DataFrame(index=daily.index)
    features["vol_vix_level"] = daily["VIX"]
    features["vol_vix_daily_change"] = daily["VIX"].diff()
    features["vol_vix_percentile_252d"] = daily["VIX"].rolling(252, min_periods=60).rank(pct=True)
    features["vol_vix_high_regime"] = (daily["VIX"] > daily["VIX"].rolling(60, min_periods=20).median()).astype(float)
    features["vol_vix9d_minus_vix"] = daily["VIX9D"] - daily["VIX"]
    features["vol_vix3m_minus_vix"] = daily["VIX3M"] - daily["VIX"]
    features["vol_vvix_level"] = daily["VVIX"]
    return _asof_wide(prediction_times, features, pd.Timedelta(days=7))


def build_market_features(
    prediction_times: pd.DatetimeIndex, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    families = []
    audits = {}
    for instrument in manifest["sources"]["mt5"]["accepted"]:
        symbol = instrument["instrument"]
        rows = _load_normalized(instrument["data_path"])
        series = rows.set_index("available_time_utc")["value"].sort_index()
        features = pd.DataFrame(index=series.index)
        features[f"market_{symbol.lower()}_return_1"] = series.pct_change()
        features[f"market_{symbol.lower()}_return_4"] = series.pct_change(4)
        features[f"market_{symbol.lower()}_volatility_16"] = series.pct_change().rolling(16, min_periods=16).std()
        aligned, audit = _asof_wide(prediction_times, features, pd.Timedelta(hours=4))
        families.append(aligned)
        audits[symbol] = audit
    if not families:
        raise RuntimeError("No MT5 instruments passed acquisition")
    result = pd.concat(families, axis=1)
    return result, {
        "instruments": audits,
        "future_observations": sum(item["future_observations"] for item in audits.values()),
    }


def build_calendar_features(
    prediction_times: pd.DatetimeIndex, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = _load_normalized(manifest["sources"]["calendar"]["data_path"])
    daily = rows.pivot(index="available_time_utc", columns="field", values="value").sort_index()
    selected = daily[
        [
            "holiday",
            "early_close",
            "day_before_holiday",
            "first_trading_day",
            "last_trading_day",
            "month_end",
            "quarter_end",
            "year_end",
        ]
    ].add_prefix("calendar_")
    aligned, audit = _asof_wide(prediction_times, selected, pd.Timedelta(days=1))
    local_times = prediction_times.tz_convert(NEW_YORK)
    early_close = aligned["calendar_early_close"].fillna(0).to_numpy() == 1
    minutes = local_times.hour * 60 + local_times.minute
    close_minutes = np.where(early_close, 13 * 60, 16 * 60)
    aligned["calendar_cash_session_exact"] = ((minutes >= 9 * 60 + 30) & (minutes < close_minutes)).astype(float)
    aligned["calendar_minutes_to_cash_close"] = close_minutes - minutes
    return aligned, audit


def directional_metrics(model: XGBoostModel, frame: pd.DataFrame) -> dict[str, Any]:
    directional = frame[frame["target"].isin([0, 1])]
    values = directional[model.feature_columns].to_numpy()
    probabilities = model.model.predict_proba(model.scaler.transform(values))
    positions = {int(label): position for position, label in enumerate(model.model.classes_)}
    score = probabilities[:, positions[1]] / np.clip(
        probabilities[:, positions[0]] + probabilities[:, positions[1]], 1e-12, None
    )
    actual = directional["target"].to_numpy(dtype=np.int8)
    predicted = (score >= 0.5).astype(np.int8)
    recalls = recall_score(actual, predicted, labels=[0, 1], average=None, zero_division=0)
    calibration = []
    bins = np.linspace(0, 1, 11)
    assignments = np.clip(np.digitize(score, bins) - 1, 0, 9)
    for index in range(10):
        selected = assignments == index
        calibration.append(
            {
                "lower": float(bins[index]),
                "upper": float(bins[index + 1]),
                "count": int(selected.sum()),
                "mean_probability": None if not selected.any() else float(score[selected].mean()),
                "observed_buy_rate": None if not selected.any() else float(actual[selected].mean()),
            }
        )
    return {
        "roc_auc": float(roc_auc_score(actual, score)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "sell_recall": float(recalls[0]),
        "buy_recall": float(recalls[1]),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "brier_score": float(brier_score_loss(actual, score)),
        "directional_rows": len(directional),
        "calibration": calibration,
    }


def _fit_model(frame: pd.DataFrame, features: list[str], parameters: dict[str, Any]) -> XGBoostModel:
    model = XGBoostModel(**parameters, models_dir=str(OUTPUT_DIR))
    model.fit_training_data(frame, features)
    if model.feature_columns != features or int(model.scaler.n_samples_seen_) != len(frame):
        raise RuntimeError("Model feature order or scaler fit rows changed")
    return model


def chronological_comparison(
    train: pd.DataFrame,
    control_features: list[str],
    treatment_features: list[str],
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for fold, (fit_indices, validation_indices) in enumerate(TimeSeriesSplit(n_splits=5).split(train), 1):
        fit = train.iloc[fit_indices]
        validation = train.iloc[validation_indices]
        control = _fit_model(fit, control_features, parameters)
        control_metrics = directional_metrics(control, validation)
        del control
        gc.collect()
        treatment = _fit_model(fit, treatment_features, parameters)
        treatment_metrics = directional_metrics(treatment, validation)
        del treatment
        gc.collect()
        rows.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(validation),
                "control_auc": control_metrics["roc_auc"],
                "treatment_auc": treatment_metrics["roc_auc"],
                "auc_gain": treatment_metrics["roc_auc"] - control_metrics["roc_auc"],
                "treatment_balanced_accuracy": treatment_metrics["balanced_accuracy"],
                "treatment_macro_f1": treatment_metrics["macro_f1"],
            }
        )
    return rows


def permutation_evidence(
    model: XGBoostModel,
    validation: pd.DataFrame,
    external_features: list[str],
) -> dict[str, Any]:
    directional = validation[validation["target"].isin([0, 1])]
    values = directional[model.feature_columns].to_numpy()
    scaled = model.scaler.transform(values)
    external_start = len(model.feature_columns) - len(external_features)
    classes = {int(label): position for position, label in enumerate(model.model.classes_)}
    actual = directional["target"].to_numpy(dtype=np.int8)

    def auc(matrix: np.ndarray) -> float:
        probabilities = model.model.predict_proba(matrix)
        score = probabilities[:, classes[1]] / np.clip(
            probabilities[:, classes[0]] + probabilities[:, classes[1]], 1e-12, None
        )
        return float(roc_auc_score(actual, score))

    observed = auc(scaled)
    random = np.random.default_rng(RANDOM_SEED)
    null_aucs = []
    for _ in range(PERMUTATIONS):
        permuted = scaled.copy()
        order = random.permutation(len(permuted))
        permuted[:, external_start:] = permuted[order, external_start:]
        null_aucs.append(auc(permuted))
    return {
        "method": "validation block permutation of all external-family columns",
        "permutations": PERMUTATIONS,
        "observed_auc": observed,
        "null_mean_auc": float(np.mean(null_aucs)),
        "null_std_auc": float(np.std(null_aucs, ddof=1)),
        "one_sided_p_value": float((1 + np.sum(np.asarray(null_aucs) >= observed)) / (PERMUTATIONS + 1)),
    }


def classify_treatment(
    validation: dict[str, Any],
    control: dict[str, Any],
    folds: list[dict[str, Any]],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    validation_gain = validation["roc_auc"] - control["roc_auc"]
    cv_gain = float(np.mean([row["auc_gain"] for row in folds]))
    positive_folds = int(sum(row["auc_gain"] > 0 for row in folds))
    credible = (
        validation["roc_auc"] >= MINIMUM_VALIDATION_AUC
        and validation["roc_auc"] > PREVIOUS_BEST_AUC
        and validation_gain >= MINIMUM_VALIDATION_GAIN
        and cv_gain >= MINIMUM_CV_GAIN
        and positive_folds >= MINIMUM_POSITIVE_FOLDS
        and permutation["one_sided_p_value"] <= 0.05
    )
    return {
        "verdict": "CREDIBLE_DIRECTIONAL_IMPROVEMENT" if credible else "RETIRE_INFORMATION_FAMILY",
        "credible": credible,
        "validation_auc_gain": validation_gain,
        "cv_mean_auc_gain": cv_gain,
        "positive_cv_folds": positive_folds,
        "required": {
            "minimum_validation_auc": MINIMUM_VALIDATION_AUC,
            "minimum_validation_gain": MINIMUM_VALIDATION_GAIN,
            "minimum_cv_gain": MINIMUM_CV_GAIN,
            "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
            "maximum_permutation_p_value": 0.05,
        },
    }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        "## Scientific Conclusion",
        f"**{report['scientific_conclusion']}**",
        "",
        "## Independent Directional Results",
        "| Family | Features | Rows (train/validation) | Control AUC | Treatment AUC | Gain | CV control | CV treatment | CV gain | Permutation p | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, result in report["treatments"].items():
        cv_control = np.mean([row["control_auc"] for row in result["chronological_cv"]])
        cv_treatment = np.mean([row["treatment_auc"] for row in result["chronological_cv"]])
        lines.append(
            f"| {name} | {result['external_feature_count']} | {result['train_rows']:,}/{result['validation_rows']:,} | "
            f"{result['control']['roc_auc']:.4f} | {result['validation']['roc_auc']:.4f} | "
            f"{result['decision']['validation_auc_gain']:+.4f} | {cv_control:.4f} | {cv_treatment:.4f} | "
            f"{result['decision']['cv_mean_auc_gain']:+.4f} | {result['permutation']['one_sided_p_value']:.4f} | {result['decision']['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Source Verdicts",
            f"- Macro: {report['source_verdicts']['macro']}",
            f"- Rates: {report['source_verdicts']['rates']}",
            f"- FX/commodities: {report['source_verdicts']['market']}",
            f"- Volatility: {report['source_verdicts']['volatility']}",
            f"- Calendar: {report['source_verdicts']['calendar']}",
            "",
            "## Integrity",
            f"- TEST rows/labels exposed: {report['integrity']['test_rows_exposed']} / {str(report['integrity']['test_labels_exposed']).lower()}",
            f"- Protected artifacts unchanged: {str(report['integrity']['protected_artifacts_unchanged']).lower()}",
            f"- Combined experiment run: {str(report['combined_experiment_run']).lower()}",
            f"- Acquired data disk: {report['resources']['acquired_data_disk_mib']:.2f} MiB",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prior experiment: {output_dir}")
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["dataset_version"] != DATASET_VERSION or not manifest["protected_artifacts_unchanged"]:
        raise RuntimeError("External-information manifest failed integrity checks")
    control_features = list(baseline_report["features"]["names_in_order"])
    if control_features != list(metadata["feature_names"]) or len(control_features) != 42:
        raise RuntimeError("Frozen 42-feature order changed")
    before = protected_hashes()
    train_raw, validation_raw, access = load_train_validation_only(
        DATASET_PATH, metadata, [*control_features, "target"]
    )
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("Sealed loader exposed TEST")
    prediction_times = pd.DatetimeIndex(
        pd.DatetimeIndex([*train_raw.index, *validation_raw.index]).tz_localize("UTC")
        + pd.Timedelta(minutes=15)
    )
    builders = {
        "rates": build_rates_features,
        "market": build_market_features,
        "volatility": build_volatility_features,
        "calendar": build_calendar_features,
    }
    family_frames = {}
    timestamp_audits = {}
    for name, builder in builders.items():
        family_frames[name], timestamp_audits[name] = builder(prediction_times, manifest)
        validate_feature_names(family_frames[name].columns)
    maximum_features = len(control_features) + max(len(frame.columns) for frame in family_frames.values())
    resources = memory_preflight(len(prediction_times), maximum_features)
    output_dir.mkdir(parents=True)
    parameters = model_parameters(baseline_report)
    frozen_control = XGBoostModel(models_dir=str(BASELINE_DIR))
    frozen_control.load(BASELINE_ID)
    if frozen_control.feature_columns != control_features:
        raise RuntimeError("Frozen model feature order changed")
    treatments = {}
    for name, external in family_frames.items():
        external_features = list(external.columns)
        feature_order = [*control_features, *external_features]
        combined = pd.concat(
            [pd.concat([train_raw, validation_raw]).set_axis(prediction_times), external], axis=1
        )
        original_train_rows = len(train_raw)
        train = combined.iloc[:original_train_rows].dropna(subset=feature_order).copy()
        validation = combined.iloc[original_train_rows:].dropna(subset=feature_order).copy()
        if train.empty or validation.empty:
            raise RuntimeError(f"{name} has no usable TRAIN/VALIDATION rows")
        control = directional_metrics(frozen_control, validation)
        model = _fit_model(train, feature_order, parameters)
        validation_metrics = directional_metrics(model, validation)
        folds = chronological_comparison(train, control_features, feature_order, parameters)
        permutation = permutation_evidence(model, validation, external_features)
        decision = classify_treatment(validation_metrics, control, folds, permutation)
        model_name = f"{EXPERIMENT_ID}_{name}"
        model_path = Path(model.save(model_name))
        reloaded = XGBoostModel(models_dir=str(output_dir))
        reloaded.load(model_name)
        sample = validation[feature_order].iloc[: min(256, len(validation))].to_numpy()
        original_prediction = model.model.predict(model.scaler.transform(sample))
        reloaded_prediction = reloaded.model.predict(reloaded.scaler.transform(sample))
        if reloaded.feature_columns != feature_order or not np.array_equal(original_prediction, reloaded_prediction):
            raise RuntimeError(f"{name} deterministic reload or feature order failed")
        treatments[name] = {
            "external_feature_count": len(external_features),
            "external_features_in_order": external_features,
            "total_feature_count": len(feature_order),
            "exact_feature_order": feature_order,
            "original_train_rows": len(train_raw),
            "original_validation_rows": len(validation_raw),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "rejected_train_rows": len(train_raw) - len(train),
            "rejected_validation_rows": len(validation_raw) - len(validation),
            "rejection_reason": "missing or stale external observation",
            "control": control,
            "validation": validation_metrics,
            "chronological_cv": folds,
            "permutation": permutation,
            "decision": decision,
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "deterministic_reload": True,
        }
        del combined, train, validation, model, reloaded
        gc.collect()
    after = protected_hashes()
    if before != after:
        raise RuntimeError("A protected artifact changed during the experiment")
    credible = [name for name, result in treatments.items() if result["decision"]["credible"]]
    conclusion = (
        "CREDIBLE_FREE_EXTERNAL_DIRECTIONAL_INFORMATION"
        if credible
        else "NO_DIRECTIONAL_SIGNAL_FROM_FREE_EXTERNAL_INFORMATION"
    )
    source_verdicts = {
        "macro": "UNAVAILABLE_RELIABLE_HISTORY; no free initial-vintage/consensus path was used",
        "rates": treatments["rates"]["decision"]["verdict"],
        "market": treatments["market"]["decision"]["verdict"],
        "volatility": treatments["volatility"]["decision"]["verdict"],
        "calendar": treatments["calendar"]["decision"]["verdict"],
    }
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_question": "Does genuinely new free information improve US30 M15 BUY-vs-SELL direction?",
        "scientific_conclusion": conclusion,
        "credible_families": credible,
        "combined_experiment_run": False,
        "combined_follow_up_eligible": credible,
        "references": {
            "chance_auc": CHANCE_AUC,
            "frozen_control_auc": 0.5129618200774144,
            "previous_best_auc": PREVIOUS_BEST_AUC,
        },
        "acquisition_manifest_path": str(MANIFEST_PATH),
        "acquisition_manifest_sha256": sha256(MANIFEST_PATH),
        "data_sources": manifest["sources"],
        "timestamp_audits": timestamp_audits,
        "treatments": treatments,
        "source_verdicts": source_verdicts,
        "resources": {
            **resources,
            "acquired_data_disk_mib": _directory_bytes(INFORMATION_ROOT) / 2**20,
            "experiment_disk_mib_before_report": _directory_bytes(output_dir) / 2**20,
        },
        "integrity": {
            "test_rows_exposed": 0,
            "test_labels_exposed": False,
            "last_exposed_timestamp": access["last_exposed_timestamp"],
            "protected_hashes_before": before,
            "protected_hashes_after": after,
            "protected_artifacts_unchanged": True,
            "exact_feature_order_verified": True,
            "deterministic_reload_verified": True,
            "future_observations": 0,
            "target_name_leakage": False,
        },
    }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "experiment_report.md").write_text(render_report(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps({"conclusion": result["scientific_conclusion"], "credible_families": result["credible_families"]}, indent=2))