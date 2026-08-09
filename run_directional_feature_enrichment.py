"""Run one controlled causal directional-context feature experiment."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from diagnose_baseline import load_train_validation_only, sha256
from diagnose_directional_signal import evaluate_saved_model
from run_ablation_25f import (
    BASELINE_DIR,
    BASELINE_ID,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DATASET_PATH,
    METADATA_PATH,
    ensure_output_available,
    model_parameters,
)
from run_first_passage_target_experiment import (
    HORIZON,
    INCOMPLETE_LABEL,
    OHLC_RECONSTRUCTION_COLUMNS,
    OUTPUT_DIR as FIRST_PASSAGE_DIR,
    TIE_LABEL,
    build_first_passage_target,
    independently_reconstruct_target,
    prepare_splits,
    reconstruct_ohlc,
)
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_directional_context_60f_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
FIRST_PASSAGE_ID = f"{BASELINE_ID}_first_passage_1atr_5bar_v1"
FIRST_PASSAGE_MODEL_PATH = FIRST_PASSAGE_DIR / f"{FIRST_PASSAGE_ID}.joblib"
FIRST_PASSAGE_REPORT_PATH = FIRST_PASSAGE_DIR / "experiment_report.json"
PREVIOUS_BEST_DIRECTION_AUC = 0.51636723758793

NEW_FEATURE_SPECS = [
    {"name": "return_lag_1", "family": "lagged_return", "lookback": 1, "sources": ["Close"], "formula": "Close.pct_change(1).shift(1)"},
    {"name": "return_lag_2", "family": "lagged_return", "lookback": 2, "sources": ["Close"], "formula": "Close.pct_change(1).shift(2)"},
    {"name": "return_lag_3", "family": "lagged_return", "lookback": 3, "sources": ["Close"], "formula": "Close.pct_change(1).shift(3)"},
    {"name": "return_lag_5", "family": "lagged_return", "lookback": 5, "sources": ["Close"], "formula": "Close.pct_change(1).shift(5)"},
    {"name": "return_20", "family": "multi_horizon_trend", "lookback": 20, "sources": ["Close"], "formula": "Close.pct_change(20)"},
    {"name": "return_40", "family": "multi_horizon_trend", "lookback": 40, "sources": ["Close"], "formula": "Close.pct_change(40)"},
    {"name": "return_80", "family": "multi_horizon_trend", "lookback": 80, "sources": ["Close"], "formula": "Close.pct_change(80)"},
    {"name": "atr_trend_5", "family": "normalized_trend", "lookback": 5, "sources": ["Close", "atr"], "formula": "(Close-Close.shift(5))/ATR[t]"},
    {"name": "atr_trend_10", "family": "normalized_trend", "lookback": 10, "sources": ["Close", "atr"], "formula": "(Close-Close.shift(10))/ATR[t]"},
    {"name": "atr_trend_20", "family": "normalized_trend", "lookback": 20, "sources": ["Close", "atr"], "formula": "(Close-Close.shift(20))/ATR[t]"},
    {"name": "atr_trend_40", "family": "normalized_trend", "lookback": 40, "sources": ["Close", "atr"], "formula": "(Close-Close.shift(40))/ATR[t]"},
    {"name": "directional_persistence_5", "family": "directional_persistence", "lookback": 5, "sources": ["Close"], "formula": "rolling_5_mean(sign(Close.pct_change()))"},
    {"name": "directional_persistence_10", "family": "directional_persistence", "lookback": 10, "sources": ["Close"], "formula": "rolling_10_mean(sign(Close.pct_change()))"},
    {"name": "directional_persistence_20", "family": "directional_persistence", "lookback": 20, "sources": ["Close"], "formula": "rolling_20_mean(sign(Close.pct_change()))"},
    {"name": "range_position_20", "family": "rolling_range_location", "lookback": 20, "sources": ["High", "Low", "Close"], "formula": "2*(Close-rolling_20_low)/(rolling_20_high-rolling_20_low)-1"},
    {"name": "range_position_50", "family": "rolling_range_location", "lookback": 50, "sources": ["High", "Low", "Close"], "formula": "2*(Close-rolling_50_low)/(rolling_50_high-rolling_50_low)-1"},
    {"name": "candle_close_location", "family": "candle_asymmetry", "lookback": 0, "sources": ["High", "Low", "Close"], "formula": "2*(Close-Low)/(High-Low)-1"},
    {"name": "signed_body_range", "family": "candle_asymmetry", "lookback": 0, "sources": ["Open", "High", "Low", "Close"], "formula": "(Close-Open)/(High-Low)"},
]
NEW_FEATURES = [spec["name"] for spec in NEW_FEATURE_SPECS]

PROTECTED_PATHS = {
    "dataset": DATASET_PATH,
    "metadata": METADATA_PATH,
    "endpoint_42f_model": BASELINE_MODEL_PATH,
    "endpoint_42f_report": BASELINE_REPORT_PATH,
    "endpoint_25f_model": BASELINE_DIR / f"{BASELINE_ID}_ablation_25f" / f"{BASELINE_ID}_ablation_25f.joblib",
    "endpoint_25f_report": BASELINE_DIR / f"{BASELINE_ID}_ablation_25f" / "experiment_report.json",
    "endpoint_18f_model": BASELINE_DIR / f"{BASELINE_ID}_ablation_18f_normalized" / f"{BASELINE_ID}_ablation_18f_normalized.joblib",
    "endpoint_18f_report": BASELINE_DIR / f"{BASELINE_ID}_ablation_18f_normalized" / "experiment_report.json",
    "first_passage_42f_model": FIRST_PASSAGE_MODEL_PATH,
    "first_passage_42f_report": FIRST_PASSAGE_REPORT_PATH,
}


def protected_hashes() -> dict[str, str]:
    return {f"{name}_sha256": sha256(path) for name, path in PROTECTED_PATHS.items()}


def _neutral_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def _centered_location(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    location = 2 * numerator.div(denominator.replace(0, np.nan)) - 1
    return location.fillna(0.0)


def build_directional_features(ohlc: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """Build the fixed causal feature manifest using current and prior rows only."""
    if not ohlc.index.equals(atr.index):
        raise ValueError("OHLC and ATR indices must match")
    if not ohlc.index.is_monotonic_increasing or ohlc.index.has_duplicates:
        raise ValueError("Directional features require unique chronological rows")
    close = ohlc["Close"]
    one_bar_return = close.pct_change()
    result = pd.DataFrame(index=ohlc.index)
    for lag in (1, 2, 3, 5):
        result[f"return_lag_{lag}"] = one_bar_return.shift(lag)
    for horizon in (20, 40, 80):
        result[f"return_{horizon}"] = close.pct_change(horizon)
    for horizon in (5, 10, 20, 40):
        result[f"atr_trend_{horizon}"] = _neutral_divide(close - close.shift(horizon), atr)
    signed_return = np.sign(one_bar_return)
    for window in (5, 10, 20):
        result[f"directional_persistence_{window}"] = signed_return.rolling(window, min_periods=window).mean()
    for window in (20, 50):
        rolling_low = ohlc["Low"].rolling(window, min_periods=window).min()
        rolling_high = ohlc["High"].rolling(window, min_periods=window).max()
        result[f"range_position_{window}"] = _centered_location(close - rolling_low, rolling_high - rolling_low)
    candle_range = ohlc["High"] - ohlc["Low"]
    result["candle_close_location"] = _centered_location(close - ohlc["Low"], candle_range)
    result["signed_body_range"] = _neutral_divide(close - ohlc["Open"], candle_range)
    if list(result.columns) != NEW_FEATURES:
        raise RuntimeError("Directional feature order differs from the fixed manifest")
    return result


def verify_feature_provenance(control_features: list[str]) -> dict[str, Any]:
    allowed_sources = {"Open", "High", "Low", "Close", "atr"}
    forbidden_tokens = ("target", "future", "label", "outcome", "lead")
    if len(control_features) != 42 or len(set(control_features)) != 42:
        raise RuntimeError("The frozen control must contain exactly 42 unique features")
    if len(NEW_FEATURES) != 18 or len(set(NEW_FEATURES)) != 18:
        raise RuntimeError("The enrichment manifest must contain exactly 18 unique features")
    if set(control_features) & set(NEW_FEATURES):
        raise RuntimeError("Enriched features must not overwrite control columns")
    for spec in NEW_FEATURE_SPECS:
        if not set(spec["sources"]).issubset(allowed_sources):
            raise RuntimeError(f"Unapproved feature source: {spec}")
        searchable = json.dumps(spec).lower()
        if any(token in searchable for token in forbidden_tokens):
            raise RuntimeError(f"Target-derived or future feature rejected: {spec['name']}")
        if int(spec["lookback"]) < 0:
            raise RuntimeError(f"Forward lookahead rejected: {spec['name']}")
    return {
        "control_feature_count": 42,
        "new_feature_count": 18,
        "total_feature_count": 60,
        "allowed_sources": sorted(allowed_sources),
        "target_derived_columns": [],
        "future_derived_columns": [],
        "all_features_causal": True,
    }


def conditional_direction_metrics(model: XGBoostModel, frame: pd.DataFrame) -> dict[str, Any]:
    directional = frame[frame["target"].isin([0, 1])]
    values = model.scaler.transform(directional[model.feature_columns].to_numpy())
    probabilities = model.model.predict_proba(values)
    positions = {int(label): position for position, label in enumerate(model.model.classes_)}
    buy = probabilities[:, positions[1]]
    sell = probabilities[:, positions[0]]
    scores = buy / np.clip(buy + sell, 1e-12, None)
    actual = directional["target"].to_numpy(dtype=int)
    predicted = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    recalls = recall_score(actual, predicted, labels=[0, 1], average=None, zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(actual, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "buy_recall": float(recalls[1]),
        "sell_recall": float(recalls[0]),
        "directional_macro_f1": float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["SELL", "BUY"],
        "directional_rows": len(directional),
        "decision_rule": "BUY when P(BUY)/(P(BUY)+P(SELL)) >= 0.5; otherwise SELL",
    }


def evaluate(model: XGBoostModel, frame: pd.DataFrame) -> dict[str, Any]:
    overall = evaluate_saved_model(model, frame)
    return {
        "primary_directional": conditional_direction_metrics(model, frame),
        "activity_vs_no_trade": overall["directional_filter"],
        "secondary_three_class": overall["multiclass"],
    }


def chronological_cv(
    train: pd.DataFrame, features: list[str], parameters: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(TimeSeriesSplit(n_splits=5).split(train), 1):
        fold_train = train.iloc[train_indices]
        fold_validation = train.iloc[validation_indices]
        model = XGBoostModel(**parameters, models_dir=str(BASELINE_DIR))
        model.fit_training_data(fold_train, features)
        metrics = evaluate(model, fold_validation)
        directional = metrics["primary_directional"]
        row = {
            "row_type": "fold",
            "fold": fold,
            **{key: directional[key] for key in ("roc_auc", "balanced_accuracy", "buy_recall", "sell_recall", "directional_macro_f1")},
            "activity_auc": metrics["activity_vs_no_trade"]["roc_auc"],
            "three_class_accuracy": metrics["secondary_three_class"]["accuracy"],
            "three_class_macro_f1": metrics["secondary_three_class"]["macro_f1"],
            "scaler_fit_rows": int(model.scaler.n_samples_seen_),
            "train_first": pd.Timestamp(fold_train.index[0]).isoformat(),
            "train_last": pd.Timestamp(fold_train.index[-1]).isoformat(),
            "validation_first": pd.Timestamp(fold_validation.index[0]).isoformat(),
            "validation_last": pd.Timestamp(fold_validation.index[-1]).isoformat(),
        }
        if row["scaler_fit_rows"] != len(fold_train):
            raise RuntimeError(f"Fold {fold} scaler row count mismatch")
        rows.append(row)
    folds = pd.DataFrame(rows)
    metric_columns = [
        "roc_auc", "balanced_accuracy", "buy_recall", "sell_recall", "directional_macro_f1",
        "activity_auc", "three_class_accuracy", "three_class_macro_f1",
    ]
    summary = {"row_type": "mean", "fold": None, **folds[metric_columns].mean().to_dict()}
    return pd.concat([folds, pd.DataFrame([summary])], ignore_index=True)


def decision(enriched: dict[str, Any], control: dict[str, Any]) -> dict[str, str]:
    auc = float(enriched["primary_directional"]["roc_auc"])
    balanced = float(enriched["primary_directional"]["balanced_accuracy"])
    buy_recall = float(enriched["primary_directional"]["buy_recall"])
    sell_recall = float(enriched["primary_directional"]["sell_recall"])
    control_auc = float(control["primary_directional"]["roc_auc"])
    comparison_auc = max(0.5, control_auc, PREVIOUS_BEST_DIRECTION_AUC)
    if auc >= 0.60 and auc >= comparison_auc + 0.05 and balanced >= 0.55 and min(buy_recall, sell_recall) >= 0.20:
        verdict = "DIRECTIONAL_SIGNAL_RECOVERED"
        recommendation = "Validate robustness on a separately authorized holdout before any deployment claim."
    elif auc <= 0.52:
        verdict = "INSUFFICIENT_DIRECTIONAL_INFORMATION"
        recommendation = (
            "Stop adding feature complexity to this M15 OHLCV representation. Next research should change the information set "
            "or horizon, such as higher-timeframe context, order-flow/microstructure data, or another instrument, in a separately approved study."
        )
    else:
        verdict = "INCONCLUSIVE"
        recommendation = "Do not claim recovered directionality; inspect temporal stability before authorizing a different research design."
    return {
        "verdict": verdict,
        "reason": (
            f"Enriched BUY-vs-SELL AUC={auc:.4f}, chance=0.5000, first-passage control={control_auc:.4f}, "
            f"previous best={PREVIOUS_BEST_DIRECTION_AUC:.4f}, balanced accuracy={balanced:.4f}, "
            f"BUY recall={buy_recall:.4f}, SELL recall={sell_recall:.4f}."
        ),
        "recommendation": recommendation,
    }


def render_report(report: dict[str, Any]) -> str:
    enriched = report["evaluation"]["enriched_validation"]
    control = report["evaluation"]["control_validation"]
    primary = enriched["primary_directional"]
    control_primary = control["primary_directional"]
    matrix = primary["confusion_matrix"]
    return "\n".join(
        [
            f"# {EXPERIMENT_ID}", "", "- Single directional feature-enrichment experiment", "- Target: unchanged five-bar first-passage +/- 1 ATR[t]", "- TEST rows/labels exposed: 0 / false", "",
            "## Primary Validation Result", "| Model | Direction AUC | Balanced accuracy | BUY recall | SELL recall | Direction macro F1 | Activity AUC |", "|---|---:|---:|---:|---:|---:|---:|",
            f"| frozen first-passage 42f | {control_primary['roc_auc']:.4f} | {control_primary['balanced_accuracy']:.4f} | {control_primary['buy_recall']:.4f} | {control_primary['sell_recall']:.4f} | {control_primary['directional_macro_f1']:.4f} | {control['activity_vs_no_trade']['roc_auc']:.4f} |",
            f"| enriched 60f | {primary['roc_auc']:.4f} | {primary['balanced_accuracy']:.4f} | {primary['buy_recall']:.4f} | {primary['sell_recall']:.4f} | {primary['directional_macro_f1']:.4f} | {enriched['activity_vs_no_trade']['roc_auc']:.4f} |", "",
            "## Conditional Direction Confusion Matrix", "Rows are actual SELL/BUY; columns are predicted SELL/BUY.", "", "```text", str(matrix[0]), str(matrix[1]), "```", "",
            "## Secondary Three-Class Context", f"- Accuracy: {enriched['secondary_three_class']['accuracy']:.4f}", f"- Balanced accuracy: {enriched['secondary_three_class']['balanced_accuracy']:.4f}", f"- Macro F1: {enriched['secondary_three_class']['macro_f1']:.4f}", "",
            "## Decision", f"**{report['decision']['verdict']}**: {report['decision']['reason']}", "", report["decision"]["recommendation"], "",
        ]
    )


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    first_passage_report = json.loads(FIRST_PASSAGE_REPORT_PATH.read_text(encoding="utf-8"))
    control_features = list(baseline_report["features"]["names_in_order"])
    if control_features != list(metadata["feature_names"]):
        raise RuntimeError("Frozen 42-feature control order differs from dataset metadata")
    provenance = verify_feature_provenance(control_features)
    features = [*control_features, *NEW_FEATURES]
    hashes_before = protected_hashes()
    requested_columns = list(dict.fromkeys([*control_features, *OHLC_RECONSTRUCTION_COLUMNS, "target"]))
    train_raw, validation_raw, access = load_train_validation_only(DATASET_PATH, metadata, requested_columns)
    exposed = pd.concat([train_raw, validation_raw])
    ohlc = reconstruct_ohlc(exposed)
    enriched_columns = build_directional_features(ohlc, exposed["atr"])
    target_rows = build_first_passage_target(ohlc, exposed["atr"])
    if not np.array_equal(target_rows["first_passage_target"].to_numpy(), independently_reconstruct_target(ohlc, exposed["atr"])):
        raise RuntimeError("First-passage target reconstruction mismatch")
    enriched = pd.concat([exposed, enriched_columns], axis=1)
    train_enriched_raw = enriched.loc[train_raw.index]
    validation_enriched_raw = enriched.loc[validation_raw.index]
    train, validation = prepare_splits(train_enriched_raw, validation_enriched_raw, target_rows)
    train = train.dropna(subset=features).copy()
    validation = validation.dropna(subset=features).copy()
    if validation.index[0] != validation_raw.index[0]:
        raise RuntimeError("Validation lost leading rows; TRAIN history should satisfy all lookbacks")
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("Sealed loader exposed TEST data")
    if int((target_rows.loc[validation_raw.index, "first_passage_target"] == INCOMPLETE_LABEL).sum()) != HORIZON:
        raise RuntimeError("Final five validation targets must remain incomplete")
    if int((target_rows["first_passage_target"] == TIE_LABEL).sum()) != int(first_passage_report["same_bar_ties"]["total"]):
        raise RuntimeError("Tie count differs from frozen first-passage control")
    if set(train["target"].unique()) != {0, 1, 2} or set(validation["target"].unique()) != {0, 1, 2}:
        raise RuntimeError("TRAIN and VALIDATION must retain all target classes")
    control_model = XGBoostModel(models_dir=str(FIRST_PASSAGE_DIR))
    control_model.load(FIRST_PASSAGE_ID)
    control_validation = evaluate(control_model, validation)
    parameters = model_parameters(baseline_report)
    model = XGBoostModel(**parameters, models_dir=str(output_dir))
    training_started = time.perf_counter()
    model.fit_training_data(train, features)
    training_seconds = time.perf_counter() - training_started
    if int(model.scaler.n_samples_seen_) != len(train) or model.feature_columns != features:
        raise RuntimeError("Final model scaler rows or exact feature order mismatch")
    evaluation = {
        "enriched_train": evaluate(model, train),
        "enriched_validation": evaluate(model, validation),
        "control_validation": control_validation,
    }
    cv_started = time.perf_counter()
    cv_results = chronological_cv(train, features, parameters)
    cv_seconds = time.perf_counter() - cv_started
    experiment_decision = decision(evaluation["enriched_validation"], control_validation)
    model_path = Path(model.save(EXPERIMENT_ID))
    reloaded = XGBoostModel(models_dir=str(output_dir))
    reloaded.load(EXPERIMENT_ID)
    values = validation[features].to_numpy()
    predictions = model.model.predict(model.scaler.transform(values))
    reloaded_predictions = reloaded.model.predict(reloaded.scaler.transform(values))
    if not np.array_equal(predictions, reloaded_predictions) or reloaded.feature_columns != features:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("Deterministic reload or exact feature order check failed")
    hashes_after = protected_hashes()
    if hashes_before != hashes_after:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("A protected dataset, model, or report changed")
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "training_seconds": training_seconds,
        "cv_seconds": cv_seconds,
        "target_definition": first_passage_report["target_definition"],
        "feature_provenance": provenance,
        "control_features": control_features,
        "new_feature_manifest": NEW_FEATURE_SPECS,
        "features_in_exact_order": features,
        "model_parameters": {**parameters, "objective": "multi:softprob", "eval_metric": "mlogloss", "early_stopping": False},
        "boundaries": {
            "original_train_rows": len(train_raw), "original_validation_rows": len(validation_raw),
            "retained_train_rows": len(train), "retained_validation_rows": len(validation),
            "train_first": pd.Timestamp(train_raw.index[0]).isoformat(), "train_last": pd.Timestamp(train_raw.index[-1]).isoformat(),
            "validation_first": pd.Timestamp(validation_raw.index[0]).isoformat(), "validation_last": pd.Timestamp(validation_raw.index[-1]).isoformat(),
            "test_first": metadata["splits"]["test"]["first_timestamp"],
        },
        "data_access": access,
        "target_integrity": {
            "independent_reconstruction_match": True,
            "same_bar_ties_excluded": int((target_rows["first_passage_target"] == TIE_LABEL).sum()),
            "incomplete_validation_tail_rows_excluded": HORIZON,
        },
        "preprocessing": {"type": "StandardScaler", "fit_scope": "retained TRAIN only", "scaler_n_samples_seen": int(model.scaler.n_samples_seen_)},
        "evaluation": evaluation,
        "cv": {"scope": "retained TRAIN only", "folds": cv_results[cv_results["row_type"] == "fold"].astype(object).where(pd.notna(cv_results), None).to_dict(orient="records"), "mean": cv_results[cv_results["row_type"] == "mean"].iloc[0].dropna().to_dict()},
        "decision_thresholds": {"minimum_auc": 0.60, "minimum_auc_gain_over_best_reference": 0.05, "minimum_balanced_accuracy": 0.55, "minimum_each_direction_recall": 0.20, "insufficient_information_auc_ceiling": 0.52},
        "decision": experiment_decision,
        "integrity": {
            "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
            "protected_artifacts_unchanged": True, "test_rows_exposed": 0, "test_labels_read": False, "test_evaluated": False,
            "reload_predictions_identical": True, "reload_feature_order_preserved": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(NEW_FEATURE_SPECS).to_csv(output_dir / "feature_manifest.csv", index=False)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    comparison = pd.DataFrame([
        {"model": "first_passage_42f_control", **control_validation["primary_directional"], "activity_auc": control_validation["activity_vs_no_trade"]["roc_auc"]},
        {"model": "directional_context_60f", **evaluation["enriched_validation"]["primary_directional"], "activity_auc": evaluation["enriched_validation"]["activity_vs_no_trade"]["roc_auc"]},
    ])
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    pd.DataFrame(evaluation["enriched_validation"]["primary_directional"]["confusion_matrix"], index=["actual_sell", "actual_buy"], columns=["predicted_sell", "predicted_buy"]).to_csv(output_dir / "directional_confusion_matrix.csv")
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "experiment_report.md").write_text(render_report(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["decision"], indent=2))