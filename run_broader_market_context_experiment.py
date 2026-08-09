"""Test causal higher-timeframe and cross-market directional context."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from acquire_directional_context_data import CONTEXT_DIR
from diagnose_baseline import load_train_validation_only, sha256
from diagnose_directional_signal import evaluate_saved_model
from run_ablation_25f import BASELINE_DIR, BASELINE_ID, BASELINE_MODEL_PATH, BASELINE_REPORT_PATH, DATASET_PATH, METADATA_PATH, ensure_output_available, model_parameters
from run_directional_feature_enrichment import NEW_FEATURES, OUTPUT_DIR as ENRICHED_DIR, build_directional_features
from run_first_passage_target_experiment import HORIZON, INCOMPLETE_LABEL, OHLC_RECONSTRUCTION_COLUMNS, OUTPUT_DIR as FIRST_PASSAGE_DIR, TIE_LABEL, build_first_passage_target, independently_reconstruct_target, prepare_splits, reconstruct_ohlc
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_broader_market_context_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
FIRST_PASSAGE_ID = f"{BASELINE_ID}_first_passage_1atr_5bar_v1"
ENRICHED_ID = f"{BASELINE_ID}_directional_context_60f_v1"
CONTEXT_MANIFEST_PATH = CONTEXT_DIR / "context_manifest.json"
TREATMENTS = ("higher_timeframe", "cross_market", "combined")
REFERENCE_AUCS = {"chance": 0.5, "previous_best": 0.51636723758793, "first_passage_42f": 0.49936062678464477, "enriched_60f": 0.49953671810381495, "enriched_60f_cv": 0.5061268643631859}

PROTECTED_PATHS = {
    "dataset": DATASET_PATH,
    "metadata": METADATA_PATH,
    "baseline_model": BASELINE_MODEL_PATH,
    "baseline_report": BASELINE_REPORT_PATH,
    "first_passage_model": FIRST_PASSAGE_DIR / f"{FIRST_PASSAGE_ID}.joblib",
    "first_passage_report": FIRST_PASSAGE_DIR / "experiment_report.json",
    "enriched_model": ENRICHED_DIR / f"{ENRICHED_ID}.joblib",
    "enriched_report": ENRICHED_DIR / "experiment_report.json",
    "context_manifest": CONTEXT_MANIFEST_PATH,
}


def protected_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    paths = dict(PROTECTED_PATHS)
    for instrument in manifest["instruments"]:
        paths[f"{instrument['instrument']}_data"] = Path(instrument["data_path"])
        paths[f"{instrument['instrument']}_provenance"] = Path(instrument["provenance_path"])
    return {f"{name}_sha256": sha256(path) for name, path in paths.items()}


def previous_baseline_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(BASELINE_DIR)).replace("\\", "/"): sha256(path)
        for path in sorted(BASELINE_DIR.rglob("*"))
        if path.is_file() and OUTPUT_DIR not in path.parents
    }


def verify_feature_names(features: list[str]) -> None:
    forbidden = ("target", "future", "label", "outcome", "lead")
    rejected = [feature for feature in features if any(token in feature.lower() for token in forbidden)]
    if rejected:
        raise RuntimeError(f"Forbidden target/future-derived context features: {rejected}")


def verify_context_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    expected_symbols = {"US500", "USTEC"}
    instruments = {row["instrument"]: row for row in manifest.get("instruments", [])}
    if set(instruments) != expected_symbols:
        raise RuntimeError("Context manifest must contain exactly US500 and USTEC")
    for symbol, row in instruments.items():
        data_path = Path(row["data_path"])
        provenance_path = Path(row["provenance_path"])
        if sha256(data_path) != row["data_sha256"] or sha256(provenance_path) != row["provenance_sha256"]:
            raise RuntimeError(f"Context hash mismatch for {symbol}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("source") != "mt5" or provenance.get("symbol") != symbol or provenance.get("timeframe") != "15m" or provenance.get("fallback_used") is not False:
            raise RuntimeError(f"Context provenance mismatch for {symbol}")
        if row.get("timezone") != "UTC" or row.get("timestamp_semantics") != "bar open" or row.get("availability_rule") != "bar_open_timestamp + 15 minutes":
            raise RuntimeError(f"Context timestamp metadata mismatch for {symbol}")
    return {"verified_symbols": sorted(instruments), "hashes_match": True, "provenance_match": True, "timezone": "UTC"}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def aggregate_completed_bars(ohlc: pd.DataFrame, rule: str, expected_bars: int) -> pd.DataFrame:
    """Aggregate M15 bars and index them by the instant their final bar closes."""
    grouped = ohlc.resample(rule, label="right", closed="left")
    bars = grouped.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    counts = grouped["Close"].count()
    return bars.loc[counts == expected_bars].dropna()


def timeframe_features(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = bars["Close"]
    returns = close.pct_change()
    ma = close.rolling(20, min_periods=20).mean()
    true_range = pd.concat([(bars["High"] - bars["Low"]), (bars["High"] - close.shift()).abs(), (bars["Low"] - close.shift()).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    rolling_high = bars["High"].rolling(20, min_periods=20).max()
    rolling_low = bars["Low"].rolling(20, min_periods=20).min()
    previous_high = bars["High"].rolling(20, min_periods=20).max().shift(1)
    previous_low = bars["Low"].rolling(20, min_periods=20).min().shift(1)
    result = pd.DataFrame(index=bars.index)
    result[f"{prefix}_return_1"] = returns
    for horizon in (2, 4, 8, 12):
        result[f"{prefix}_return_{horizon}"] = close.pct_change(horizon)
    result[f"{prefix}_trend_direction"] = np.sign(close - ma)
    result[f"{prefix}_ma_slope"] = _safe_divide(ma - ma.shift(4), atr)
    result[f"{prefix}_distance_ma"] = _safe_divide(close - ma, atr)
    result[f"{prefix}_volatility_regime"] = _safe_divide(atr, atr.rolling(40, min_periods=40).median())
    result[f"{prefix}_range_position"] = 2 * _safe_divide(close - rolling_low, rolling_high - rolling_low) - 1
    result[f"{prefix}_breakout_state"] = np.select([close > previous_high, close < previous_low], [1.0, -1.0], default=0.0)
    result[f"{prefix}_persistence"] = np.sign(returns).rolling(8, min_periods=8).mean()
    return result


def asof_features(prediction_times: pd.DatetimeIndex, features: pd.DataFrame, prefix: str, max_age: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_timezone = prediction_times.tz
    source_timezone = features.index.tz
    if prediction_timezone != source_timezone:
        raise RuntimeError(
            f"Timezone mismatch: prediction={prediction_timezone}, source={source_timezone}"
        )
    if not prediction_times.is_monotonic_increasing or not features.index.is_monotonic_increasing:
        raise RuntimeError("Point-in-time joins require chronological timestamps")
    prediction_times = prediction_times.as_unit("ns")
    source = features.copy()
    source.index = source.index.as_unit("ns")
    source["source_timestamp"] = source.index
    left = pd.DataFrame({"prediction_timestamp": prediction_times}, index=prediction_times)
    merged = pd.merge_asof(left.reset_index(drop=True).sort_values("prediction_timestamp"), source.reset_index(drop=True).sort_values("source_timestamp"), left_on="prediction_timestamp", right_on="source_timestamp", direction="backward", tolerance=max_age)
    merged.index = prediction_times
    feature_columns = [column for column in features.columns]
    merged_features = merged[feature_columns].add_prefix(f"{prefix}_") if prefix else merged[feature_columns]
    audit = pd.DataFrame({"prediction_timestamp": prediction_times, "source_timestamp": merged["source_timestamp"].to_numpy()}, index=prediction_times)
    audit["source_age_minutes"] = (audit["prediction_timestamp"] - audit["source_timestamp"]).dt.total_seconds() / 60
    return merged_features, audit


def build_higher_timeframe_context(ohlc: pd.DataFrame, prediction_times: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    families = []
    audits = {}
    for prefix, rule, expected, age in (("h1", "1h", 4, "8h"), ("h4", "4h", 16, "24h")):
        bars = aggregate_completed_bars(ohlc, rule, expected)
        features = timeframe_features(bars, prefix)
        aligned, audit = asof_features(prediction_times, features, "", pd.Timedelta(age))
        families.append(aligned)
        audits[prefix] = audit
    return pd.concat(families, axis=1), audits


def market_features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    close = frame["Close"]
    returns = close.pct_change()
    result = pd.DataFrame(index=frame.index + pd.Timedelta(minutes=15))
    result[f"{symbol}_return_1"] = returns.to_numpy()
    for horizon in (2, 4, 8, 12, 24):
        result[f"{symbol}_return_{horizon}"] = close.pct_change(horizon).to_numpy()
    result[f"{symbol}_momentum_acceleration"] = (close.pct_change(4) - close.pct_change(4).shift(4)).to_numpy()
    result[f"{symbol}_persistence"] = np.sign(returns).rolling(12, min_periods=12).mean().to_numpy()
    result[f"{symbol}_realized_volatility"] = returns.rolling(24, min_periods=24).std().to_numpy()
    ranges = (frame["High"] - frame["Low"]) / close
    result[f"{symbol}_range_regime"] = _safe_divide(ranges.rolling(12, min_periods=12).mean(), ranges.rolling(96, min_periods=96).median()).to_numpy()
    return result


def build_cross_market_context(ohlc: pd.DataFrame, prediction_times: pd.DatetimeIndex, manifest: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    families = []
    audits = {}
    for instrument in manifest["instruments"]:
        symbol = instrument["instrument"]
        raw = pd.read_parquet(instrument["data_path"])
        features = market_features(raw, symbol)
        aligned, audit = asof_features(prediction_times, features, "", pd.Timedelta(hours=4))
        families.append(aligned)
        audits[symbol] = audit
    context = pd.concat(families, axis=1)
    us30_return_4 = ohlc["Close"].pct_change(4).set_axis(prediction_times)
    us30_return_12 = ohlc["Close"].pct_change(12).set_axis(prediction_times)
    for symbol in ("US500", "USTEC"):
        context[f"US30_minus_{symbol}_return_4"] = us30_return_4 - context[f"{symbol}_return_4"]
        context[f"US30_minus_{symbol}_return_12"] = us30_return_12 - context[f"{symbol}_return_12"]
        context[f"US30_{symbol}_alignment"] = np.sign(us30_return_4) * np.sign(context[f"{symbol}_return_4"])
    context["US500_USTEC_alignment"] = np.sign(context["US500_return_4"]) * np.sign(context["USTEC_return_4"])
    context["US500_minus_USTEC_return_4"] = context["US500_return_4"] - context["USTEC_return_4"]
    return context, audits


def verify_timestamp_audits(audits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary = {}
    for family, audit in audits.items():
        valid = audit.dropna(subset=["source_timestamp"])
        if (valid["source_timestamp"] > valid["prediction_timestamp"]).any():
            raise RuntimeError(f"Future observation detected in {family}")
        summary[family] = {"rows": len(audit), "matched_rows": len(valid), "max_source_age_minutes": float(valid["source_age_minutes"].max()), "future_observations": 0}
    return summary


def conditional_metrics(model: XGBoostModel, frame: pd.DataFrame) -> dict[str, Any]:
    directional = frame[frame["target"].isin([0, 1])]
    probabilities = model.model.predict_proba(model.scaler.transform(directional[model.feature_columns].to_numpy()))
    positions = {int(label): position for position, label in enumerate(model.model.classes_)}
    buy = probabilities[:, positions[1]]
    sell = probabilities[:, positions[0]]
    score = buy / np.clip(buy + sell, 1e-12, None)
    actual = directional["target"].to_numpy(dtype=int)
    predicted = (score >= 0.5).astype(int)
    recalls = recall_score(actual, predicted, labels=[0, 1], average=None, zero_division=0)
    precisions = precision_score(actual, predicted, labels=[0, 1], average=None, zero_division=0)
    bins = pd.cut(score, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame({"score": score, "actual": actual}).groupby(bins, observed=False).agg(mean_score=("score", "mean"), observed_buy_rate=("actual", "mean"), count=("actual", "size")).reset_index(drop=True)
    return {"roc_auc": float(roc_auc_score(actual, score)), "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)), "buy_recall": float(recalls[1]), "sell_recall": float(recalls[0]), "buy_precision": float(precisions[1]), "sell_precision": float(precisions[0]), "directional_macro_f1": float(f1_score(actual, predicted, average="macro")), "brier_score": float(brier_score_loss(actual, score)), "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(), "directional_rows": len(directional), "calibration": calibration.astype(object).where(pd.notna(calibration), None).to_dict(orient="records")}


def evaluate(model: XGBoostModel, frame: pd.DataFrame) -> dict[str, Any]:
    overall = evaluate_saved_model(model, frame)
    return {"directional": conditional_metrics(model, frame), "activity_auc": overall["directional_filter"]["roc_auc"], "three_class": overall["multiclass"]}


def fit_and_cv(name: str, train: pd.DataFrame, validation: pd.DataFrame, features: list[str], parameters: dict[str, Any], output_dir: Path) -> tuple[XGBoostModel, dict[str, Any], pd.DataFrame]:
    model = XGBoostModel(**parameters, models_dir=str(output_dir))
    model.fit_training_data(train, features)
    if int(model.scaler.n_samples_seen_) != len(train) or model.feature_columns != features:
        raise RuntimeError(f"{name} scaler rows or feature order mismatch")
    evaluation = {"train": evaluate(model, train), "validation": evaluate(model, validation)}
    rows = []
    for fold, (train_idx, validation_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(train), 1):
        fold_train, fold_validation = train.iloc[train_idx], train.iloc[validation_idx]
        fold_model = XGBoostModel(**parameters, models_dir=str(output_dir))
        fold_model.fit_training_data(fold_train, features)
        metrics = evaluate(fold_model, fold_validation)
        rows.append({"treatment": name, "fold": fold, "direction_auc": metrics["directional"]["roc_auc"], "balanced_accuracy": metrics["directional"]["balanced_accuracy"], "buy_recall": metrics["directional"]["buy_recall"], "sell_recall": metrics["directional"]["sell_recall"], "directional_macro_f1": metrics["directional"]["directional_macro_f1"], "activity_auc": metrics["activity_auc"], "scaler_fit_rows": int(fold_model.scaler.n_samples_seen_)})
        if rows[-1]["scaler_fit_rows"] != len(fold_train):
            raise RuntimeError(f"{name} fold {fold} scaler row mismatch")
    return model, evaluation, pd.DataFrame(rows)


def classify_result(results: dict[str, Any], cv: pd.DataFrame) -> dict[str, str]:
    best_name = max(results, key=lambda name: results[name]["validation"]["directional"]["roc_auc"])
    best = results[best_name]["validation"]
    auc = best["directional"]["roc_auc"]
    cv_auc = float(cv.loc[cv["treatment"] == best_name, "direction_auc"].mean())
    if auc >= 0.60 and cv_auc >= 0.56:
        verdict = "DIRECTIONAL_SIGNAL_RECOVERED"
    elif auc >= 0.54 and cv_auc >= 0.52:
        verdict = "PARTIAL_DIRECTIONAL_SIGNAL"
    else:
        verdict = "FILTER_ONLY_SIGNAL_CONFIRMED"
    return {"verdict": verdict, "best_treatment": best_name, "reason": f"Best validation directional AUC={auc:.4f}; its five-fold mean directional AUC={cv_auc:.4f}; activity AUC={best['activity_auc']:.4f}."}


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(CONTEXT_MANIFEST_PATH.read_text(encoding="utf-8"))
    context_verification = verify_context_manifest(manifest)
    hashes_before = protected_hashes(manifest)
    previous_hashes_before = previous_baseline_hashes()
    control_features = list(baseline_report["features"]["names_in_order"])
    requested = list(dict.fromkeys([*control_features, *OHLC_RECONSTRUCTION_COLUMNS, "target"]))
    train_raw, validation_raw, access = load_train_validation_only(DATASET_PATH, metadata, requested)
    exposed = pd.concat([train_raw, validation_raw])
    ohlc = reconstruct_ohlc(exposed)
    target_rows = build_first_passage_target(ohlc, exposed["atr"])
    if not np.array_equal(target_rows["first_passage_target"].to_numpy(), independently_reconstruct_target(ohlc, exposed["atr"])):
        raise RuntimeError("First-passage target reconstruction mismatch")
    prediction_times = pd.DatetimeIndex(exposed.index + pd.Timedelta(minutes=15))
    ht, ht_audits = build_higher_timeframe_context(ohlc, prediction_times)
    cross, cross_audits = build_cross_market_context(ohlc, prediction_times, manifest)
    enriched_m15 = build_directional_features(ohlc, exposed["atr"]).set_axis(prediction_times)
    timestamp_audit = verify_timestamp_audits({**ht_audits, **cross_audits})
    context = pd.concat([exposed.set_axis(prediction_times), enriched_m15, ht, cross], axis=1)
    train_context_raw = context.iloc[:len(train_raw)].copy()
    validation_context_raw = context.iloc[len(train_raw):].copy()
    target_for_decision = target_rows.set_axis(prediction_times)
    train_all, validation_all = prepare_splits(train_context_raw, validation_context_raw, target_for_decision)
    ht_features, cross_features = list(ht.columns), list(cross.columns)
    all_context_features = [*ht_features, *cross_features]
    common_required = [*control_features, *all_context_features]
    train = train_all.dropna(subset=common_required).copy()
    validation = validation_all.dropna(subset=common_required).copy()
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"] or int((target_rows.loc[validation_raw.index, "first_passage_target"] == INCOMPLETE_LABEL).sum()) != HORIZON:
        raise RuntimeError("TEST seal or validation-tail target guard failed")
    verify_feature_names(all_context_features)
    feature_groups = {"higher_timeframe": [*control_features, *ht_features], "cross_market": [*control_features, *cross_features], "combined": [*control_features, *ht_features, *cross_features]}
    parameters = model_parameters(baseline_report)
    control_a = XGBoostModel(models_dir=str(FIRST_PASSAGE_DIR)); control_a.load(FIRST_PASSAGE_ID)
    control_b = XGBoostModel(models_dir=str(ENRICHED_DIR)); control_b.load(ENRICHED_ID)
    controls = {
        "control_a_first_passage_42f": evaluate(control_a, validation),
        "control_b_enriched_60f": evaluate(control_b, validation),
    }
    if control_a.feature_columns != control_features or control_b.feature_columns != [*control_features, *NEW_FEATURES]:
        raise RuntimeError("Frozen control feature order mismatch")
    output_dir.mkdir(parents=True)
    models, results, cv_frames = {}, {}, []
    for treatment in TREATMENTS:
        model, treatment_results, folds = fit_and_cv(treatment, train, validation, feature_groups[treatment], parameters, output_dir)
        model.save(f"{EXPERIMENT_ID}_{treatment}")
        models[treatment], results[treatment] = model, treatment_results
        cv_frames.append(folds)
    cv = pd.concat(cv_frames, ignore_index=True)
    for treatment, model in models.items():
        reloaded = XGBoostModel(models_dir=str(output_dir)); reloaded.load(f"{EXPERIMENT_ID}_{treatment}")
        values = validation[feature_groups[treatment]].to_numpy()
        if not np.array_equal(model.model.predict(model.scaler.transform(values)), reloaded.model.predict(reloaded.scaler.transform(values))) or reloaded.feature_columns != feature_groups[treatment]:
            raise RuntimeError(f"{treatment} deterministic reload failed")
    hashes_after = protected_hashes(manifest)
    previous_hashes_after = {
        relative: sha256(BASELINE_DIR / relative)
        for relative in previous_hashes_before
    }
    if hashes_before != hashes_after or previous_hashes_before != previous_hashes_after:
        raise RuntimeError("Protected artifact changed")
    decision = classify_result(results, cv)
    report = {"experiment_id": EXPERIMENT_ID, "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.perf_counter() - started, "target_definition": {"horizon_bars": 5, "horizon_minutes": 75, "barrier": "+/- 1 ATR[t]", "first_passage": True, "unresolved": "NO_TRADE", "same_bar_ties": "excluded", "incomplete_validation_tail": 5}, "data_provenance": manifest, "context_verification": context_verification, "unavailable_families": manifest["unavailable_families"], "timestamp_semantics": {"base_bar_timestamp": "M15 bar open UTC", "prediction_timestamp": "base bar open + 15 minutes", "higher_timeframe_availability": "aggregate right-edge close <= prediction timestamp", "cross_market_availability": "external M15 bar open + 15 minutes <= prediction timestamp", "timestamp_audit": timestamp_audit}, "feature_groups": {name: {"count": len(features), "exact_order": features} for name, features in feature_groups.items()}, "boundaries": {"original_train_rows": len(train_raw), "original_validation_rows": len(validation_raw), "common_retained_train_rows": len(train), "common_retained_validation_rows": len(validation), "last_validation_prediction_timestamp": pd.Timestamp(validation.index[-1]).isoformat(), "test_rows_exposed": 0}, "model_parameters": {**parameters, "objective": "multi:softprob", "eval_metric": "mlogloss", "early_stopping": False}, "references": REFERENCE_AUCS, "controls_on_common_validation": controls, "results": results, "cv": {"folds": cv.to_dict(orient="records"), "means": cv.groupby("treatment")[["direction_auc", "balanced_accuracy", "buy_recall", "sell_recall", "directional_macro_f1", "activity_auc"]].mean().reset_index().to_dict(orient="records")}, "decision": decision, "scientific_answer": "Broader market context recovered directional information." if decision["verdict"] == "DIRECTIONAL_SIGNAL_RECOVERED" else "Broader market context did not provide robust directional information absent from US30 M15 OHLCV; the reproducible signal remains primarily the likelihood of an active move.", "integrity": {"protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after, "protected_artifacts_checked": len(previous_hashes_before), "protected_artifacts_unchanged": True, "test_rows_exposed": 0, "test_labels_read": False, "test_evaluated": False, "all_reload_predictions_identical": True, "all_feature_orders_preserved": True}}
    cv.to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame([{"treatment": name, **values["validation"]["directional"], "activity_auc": values["validation"]["activity_auc"], "three_class_accuracy": values["validation"]["three_class"]["accuracy"], "three_class_macro_f1": values["validation"]["three_class"]["macro_f1"]} for name, values in results.items()]).to_csv(output_dir / "validation_comparison.csv", index=False)
    pd.DataFrame([{"family": family, **values} for family, values in timestamp_audit.items()]).to_csv(output_dir / "timestamp_audit.csv", index=False)
    row_audits = []
    for family, audit in {**ht_audits, **cross_audits}.items():
        family_audit = audit.copy()
        family_audit.insert(0, "feature_family", family)
        row_audits.append(family_audit)
    pd.concat(row_audits).to_parquet(output_dir / "feature_source_timestamps.parquet")
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"# {EXPERIMENT_ID}", "", "## Decision", f"**{decision['verdict']}**: {decision['reason']}", "", "## Scientific Answer", report["scientific_answer"], "", "## Validation Directional AUC"] + [f"- {name}: {results[name]['validation']['directional']['roc_auc']:.4f} (activity AUC {results[name]['validation']['activity_auc']:.4f})" for name in TREATMENTS]
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment()["decision"], indent=2))