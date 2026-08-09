"""Test whether M1 structure or forecast horizon recovers US30 directionality."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acquire_microstructure_history import DATA_PATH as M1_DATA_PATH
from acquire_microstructure_history import MANIFEST_PATH as M1_MANIFEST_PATH
from acquire_microstructure_history import PROVENANCE_PATH as M1_PROVENANCE_PATH
from diagnose_baseline import load_train_validation_only, sha256
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
from run_broader_market_context_experiment import evaluate, fit_and_cv
from run_first_passage_target_experiment import (
    INCOMPLETE_LABEL,
    OHLC_RECONSTRUCTION_COLUMNS,
    OUTPUT_DIR as FIRST_PASSAGE_DIR,
    TIE_LABEL,
    build_first_passage_target,
    reconstruct_ohlc,
)
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_microstructure_horizon_directional_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
FIRST_PASSAGE_ID = f"{BASELINE_ID}_first_passage_1atr_5bar_v1"
FIRST_PASSAGE_REPORT_PATH = FIRST_PASSAGE_DIR / "experiment_report.json"
HORIZONS = (1, 2, 3, 4, 5, 8)
REFERENCE_BEST_AUC = 0.51636723758793
DECISION_THRESHOLDS = {
    "weak_auc": 0.52,
    "promising_validation_auc": 0.54,
    "promising_cv_auc": 0.52,
    "promising_gain_over_previous_best": 0.02,
    "robust_validation_auc": 0.58,
    "robust_cv_auc": 0.55,
    "robust_gain_over_previous_best": 0.04,
    "robust_max_cv_std": 0.04,
}

M1_PRICE_FEATURES = [
    "m1_block_return",
    "m1_direction_imbalance",
    "m1_bullish_fraction",
    "m1_return_mean",
    "m1_return_std",
    "m1_return_skew",
    "m1_range_expansion",
    "m1_wick_imbalance",
    "m1_range_position",
    "m1_realized_volatility",
    "m1_volatility_acceleration",
    "m1_max_bull_run",
    "m1_max_bear_run",
    "m1_momentum_5",
    "m1_reversal_structure",
    "m1_distance_from_open",
    "m1_distance_from_high",
    "m1_distance_from_low",
    "m1_new_high_fraction",
    "m1_new_low_fraction",
    "m1_trend_slope",
    "m1_return_autocorrelation",
    "m1_range_atr_ratio",
    "m1_tick_volume_mean",
    "m1_tick_volume_change",
    "m1_signed_tick_volume_imbalance",
]
SPREAD_FEATURES = [
    "m1_spread_mean",
    "m1_spread_std",
    "m1_spread_max",
    "m1_spread_last",
    "m1_spread_change",
]


def _max_run(values: np.ndarray, direction: int) -> int:
    best = current = 0
    for value in values:
        if int(value) == direction:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _block_features(block: pd.DataFrame) -> dict[str, float]:
    close = block["Close"].to_numpy(dtype=float)
    open_price = block["Open"].to_numpy(dtype=float)
    high = block["High"].to_numpy(dtype=float)
    low = block["Low"].to_numpy(dtype=float)
    tick_volume = block["TickVolume"].to_numpy(dtype=float)
    spread = block["Spread"].to_numpy(dtype=float)
    returns = np.diff(close) / close[:-1]
    directions = np.sign(close - open_price)
    block_range = float(high.max() - low.min())
    upper_wicks = high - np.maximum(open_price, close)
    lower_wicks = np.minimum(open_price, close) - low
    x = np.arange(len(close), dtype=float)
    slope = float(np.polyfit(x, close, 1)[0]) if block_range else 0.0
    autocorrelation = (
        float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        if len(returns) > 2 and np.std(returns[:-1]) > 0 and np.std(returns[1:]) > 0
        else 0.0
    )
    first_return = close[6] / close[0] - 1
    second_return = close[-1] / close[7] - 1
    signed_volume = float(np.sum(directions * tick_volume))
    return {
        "_m1_block_range": block_range,
        "m1_block_return": float(close[-1] / open_price[0] - 1),
        "m1_direction_imbalance": float(directions.mean()),
        "m1_bullish_fraction": float((directions > 0).mean()),
        "m1_return_mean": float(returns.mean()),
        "m1_return_std": float(returns.std(ddof=0)),
        "m1_return_skew": float(pd.Series(returns).skew()) if np.std(returns) else 0.0,
        "m1_wick_imbalance": _safe_ratio(float(lower_wicks.sum() - upper_wicks.sum()), block_range),
        "m1_range_position": _safe_ratio(float(2 * close[-1] - high.max() - low.min()), block_range),
        "m1_realized_volatility": float(np.sqrt(np.sum(np.square(returns)))),
        "m1_max_bull_run": float(_max_run(directions, 1)),
        "m1_max_bear_run": float(_max_run(directions, -1)),
        "m1_momentum_5": float(close[-1] / close[-6] - 1),
        "m1_reversal_structure": float(second_return - first_return),
        "m1_distance_from_open": _safe_ratio(float(close[-1] - open_price[0]), block_range),
        "m1_distance_from_high": _safe_ratio(float(close[-1] - high.max()), block_range),
        "m1_distance_from_low": _safe_ratio(float(close[-1] - low.min()), block_range),
        "m1_new_high_fraction": float(np.mean(high == np.maximum.accumulate(high))),
        "m1_new_low_fraction": float(np.mean(low == np.minimum.accumulate(low))),
        "m1_trend_slope": _safe_ratio(slope, block_range),
        "m1_return_autocorrelation": autocorrelation,
        "m1_tick_volume_mean": float(tick_volume.mean()),
        "m1_signed_tick_volume_imbalance": _safe_ratio(signed_volume, float(tick_volume.sum())),
        "m1_spread_mean": float(spread.mean()),
        "m1_spread_std": float(spread.std(ddof=0)),
        "m1_spread_max": float(spread.max()),
        "m1_spread_last": float(spread[-1]),
    }


def build_m1_features(
    m1: pd.DataFrame, prediction_times: pd.DatetimeIndex, atr: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate only complete M1 bars whose closes are at or before prediction."""
    required = {"Open", "High", "Low", "Close", "TickVolume", "Spread", "RealVolume"}
    if not required.issubset(m1.columns):
        raise RuntimeError("M1 input does not contain the validated source schema")
    if m1.index.tz is not None or prediction_times.tz is not None:
        raise RuntimeError("M1 and prediction timestamps must both be UTC-naive")
    source = m1.copy()
    source["source_timestamp"] = source.index + pd.Timedelta(minutes=1)
    source["prediction_timestamp"] = source.index.floor("15min") + pd.Timedelta(minutes=15)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for prediction_timestamp, block in source.groupby("prediction_timestamp", sort=True):
        if len(block) != 15:
            continue
        maximum_source = pd.Timestamp(block["source_timestamp"].max())
        if maximum_source > prediction_timestamp:
            raise RuntimeError("M1 feature block contains a post-prediction observation")
        rows.append({"prediction_timestamp": prediction_timestamp, **_block_features(block)})
        audits.append(
            {
                "prediction_timestamp": prediction_timestamp,
                "maximum_source_timestamp": maximum_source,
                "source_rows": len(block),
                "source_not_after_prediction": True,
            }
        )
    features = pd.DataFrame(rows).set_index("prediction_timestamp")
    audit = pd.DataFrame(audits).set_index("prediction_timestamp")
    features["m1_range_expansion"] = (
        features["m1_realized_volatility"]
        / features["m1_realized_volatility"].rolling(20, min_periods=20).median().replace(0, np.nan)
    )
    features["m1_volatility_acceleration"] = (
        features["m1_realized_volatility"]
        / features["m1_realized_volatility"].shift(4).replace(0, np.nan)
    )
    features["m1_tick_volume_change"] = (
        features["m1_tick_volume_mean"]
        / features["m1_tick_volume_mean"].shift(1).replace(0, np.nan)
        - 1
    )
    features["m1_spread_change"] = features["m1_spread_last"].diff()
    aligned_atr = atr.copy()
    aligned_atr.index = prediction_times
    m15_range = aligned_atr.reindex(features.index)
    features["m1_range_atr_ratio"] = features.pop("_m1_block_range") / m15_range.replace(0, np.nan)
    features = features.reindex(prediction_times)
    audit = audit.reindex(prediction_times)
    ordered = [*M1_PRICE_FEATURES, *SPREAD_FEATURES]
    if set(features.columns) != set(ordered):
        raise RuntimeError("M1 feature manifest and generated columns differ")
    return features[ordered], audit


def verify_m1_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    m1 = manifest.get("m1", {})
    if manifest.get("version") != "microstructure_mt5_v1":
        raise RuntimeError("Unexpected M1 manifest version")
    if m1.get("timeframe") != "M1" or m1.get("timezone") != "UTC":
        raise RuntimeError("M1 timeframe/timezone provenance mismatch")
    if m1.get("availability_rule") != "bar_open_timestamp + 1 minute":
        raise RuntimeError("M1 availability rule mismatch")
    if sha256(M1_DATA_PATH) != m1.get("data_sha256"):
        raise RuntimeError("M1 data hash mismatch")
    if sha256(M1_PROVENANCE_PATH) != m1.get("provenance_sha256"):
        raise RuntimeError("M1 provenance hash mismatch")
    unavailable = manifest.get("unavailable_families", {})
    if unavailable.get("historical_ticks", {}).get("status") != "UNAVAILABLE_RELIABLE_HISTORY":
        raise RuntimeError("Historical tick availability was not resolved")
    return {
        "data_hash_verified": True,
        "provenance_hash_verified": True,
        "historical_ticks": "UNAVAILABLE_RELIABLE_HISTORY",
    }


def protected_hashes() -> dict[str, str]:
    paths = {
        "dataset": DATASET_PATH,
        "metadata": METADATA_PATH,
        "baseline_model": BASELINE_MODEL_PATH,
        "baseline_report": BASELINE_REPORT_PATH,
        "first_passage_model": FIRST_PASSAGE_DIR / f"{FIRST_PASSAGE_ID}.joblib",
        "first_passage_report": FIRST_PASSAGE_REPORT_PATH,
        "m1_data": M1_DATA_PATH,
        "m1_provenance": M1_PROVENANCE_PATH,
        "m1_manifest": M1_MANIFEST_PATH,
    }
    return {f"{name}_sha256": sha256(path) for name, path in paths.items()}


def previous_experiment_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(BASELINE_DIR)).replace("\\", "/"): sha256(path)
        for path in sorted(BASELINE_DIR.rglob("*"))
        if path.is_file() and OUTPUT_DIR not in path.parents
    }


def classify_candidate(validation_auc: float, cv_auc: float, cv_std: float) -> str:
    gain = validation_auc - REFERENCE_BEST_AUC
    if (
        validation_auc >= DECISION_THRESHOLDS["robust_validation_auc"]
        and cv_auc >= DECISION_THRESHOLDS["robust_cv_auc"]
        and gain >= DECISION_THRESHOLDS["robust_gain_over_previous_best"]
        and cv_std <= DECISION_THRESHOLDS["robust_max_cv_std"]
    ):
        return "ROBUST_DIRECTIONAL_SIGNAL"
    if (
        validation_auc >= DECISION_THRESHOLDS["promising_validation_auc"]
        and cv_auc >= DECISION_THRESHOLDS["promising_cv_auc"]
        and gain >= DECISION_THRESHOLDS["promising_gain_over_previous_best"]
    ):
        return "PROMISING_DIRECTIONAL_SIGNAL"
    if validation_auc >= DECISION_THRESHOLDS["weak_auc"] or cv_auc >= DECISION_THRESHOLDS["weak_auc"]:
        return "WEAK_UNCONFIRMED_SIGNAL"
    return "NO_DIRECTIONAL_SIGNAL"


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["target"].value_counts().reindex([0, 1, 2], fill_value=0)
    return {
        "rows": len(frame),
        "sell": int(counts[0]),
        "buy": int(counts[1]),
        "no_trade": int(counts[2]),
        "sell_pct": float(100 * counts[0] / len(frame)),
        "buy_pct": float(100 * counts[1] / len(frame)),
        "no_trade_pct": float(100 * counts[2] / len(frame)),
    }


def economic_sanity_check(
    model: XGBoostModel,
    validation: pd.DataFrame,
    features: list[str],
    ohlc: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    values = validation[features].to_numpy()
    predictions = model.model.predict(model.scaler.transform(values)).astype(int)
    active = predictions != 2
    active_index = validation.index[active]
    future_close = ohlc["Close"].shift(-horizon).reindex(active_index)
    entry_close = ohlc["Close"].reindex(active_index)
    directions = np.where(predictions[active] == 1, 1.0, -1.0)
    signed_returns = directions * (future_close.to_numpy() / entry_close.to_numpy() - 1)
    actual = validation.loc[active_index, "target"].to_numpy(dtype=int)
    directional_actual = np.isin(actual, [0, 1])
    accuracy = float(
        np.mean(predictions[active][directional_actual] == actual[directional_actual])
    ) if directional_actual.any() else None
    spread_points = validation.loc[active_index, "m1_spread_mean"].to_numpy(dtype=float) * 0.01
    spread_only_returns = signed_returns - spread_points / entry_close.to_numpy()
    return {
        "performed": True,
        "framework": "All model-predicted BUY/SELL signals; signed endpoint return at the treatment horizon.",
        "signal_count": int(active.sum()),
        "directional_accuracy_on_realized_buy_sell": accuracy,
        "average_signed_move_after_prediction": float(np.nanmean(signed_returns)),
        "expected_return_before_costs": float(np.nanmean(signed_returns)),
        "expected_return_after_observed_m1_spread_only": float(np.nanmean(spread_only_returns)),
        "observed_spread_assumption": "M1 bar spread in points multiplied by verified point size 0.01.",
        "historical_slippage_available": False,
        "expected_return_after_conservative_spread_and_slippage": "UNAVAILABLE_RELIABLE_HISTORY",
    }


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    first_passage_report = json.loads(FIRST_PASSAGE_REPORT_PATH.read_text(encoding="utf-8"))
    m1_manifest = json.loads(M1_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_verification = verify_m1_manifest(m1_manifest)
    hashes_before = protected_hashes()
    experiments_before = previous_experiment_hashes()
    control_features = list(baseline_report["features"]["names_in_order"])
    if control_features != list(metadata["feature_names"]) or len(control_features) != 42:
        raise RuntimeError("Frozen 42-feature order mismatch")
    requested = list(dict.fromkeys([*control_features, *OHLC_RECONSTRUCTION_COLUMNS, "target"]))
    train_raw, validation_raw, access = load_train_validation_only(
        DATASET_PATH, metadata, requested
    )
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("Sealed loader exposed TEST")
    exposed = pd.concat([train_raw, validation_raw])
    prediction_times = pd.DatetimeIndex(exposed.index + pd.Timedelta(minutes=15))
    ohlc = reconstruct_ohlc(exposed)
    ohlc.index = prediction_times
    atr = exposed["atr"].copy()
    m1 = pd.read_parquet(M1_DATA_PATH)
    if m1.index[-1] >= pd.Timestamp(metadata["splits"]["test"]["first_timestamp"]):
        raise RuntimeError("M1 artifact reaches the TEST boundary")
    m1_features, timestamp_audit = build_m1_features(m1, prediction_times, atr)
    if (timestamp_audit["maximum_source_timestamp"] > timestamp_audit.index).fillna(False).any():
        raise RuntimeError("Post-prediction M1 source detected")
    base = exposed.copy()
    base.index = prediction_times
    context = pd.concat([base, m1_features], axis=1)
    parameters = model_parameters(baseline_report)
    output_dir.mkdir(parents=True)
    frozen_control = XGBoostModel(models_dir=str(FIRST_PASSAGE_DIR))
    frozen_control.load(FIRST_PASSAGE_ID)
    if frozen_control.feature_columns != control_features:
        raise RuntimeError("Frozen first-passage model feature order mismatch")
    models: dict[str, XGBoostModel] = {}
    results: dict[str, Any] = {}
    cv_frames: list[pd.DataFrame] = []
    distributions: dict[str, Any] = {}
    retained_frames: dict[str, pd.DataFrame] = {}
    feature_groups: dict[str, list[str]] = {}
    for horizon in HORIZONS:
        target_rows = build_first_passage_target(ohlc, base["atr"], horizon=horizon)
        if int((target_rows["first_passage_target"] == INCOMPLETE_LABEL).sum()) != horizon:
            raise RuntimeError(f"Horizon {horizon} validation-tail guard failed")
        labeled = context.copy()
        labeled["target"] = target_rows["first_passage_target"]
        train_labeled = labeled.iloc[: len(train_raw)]
        validation_labeled = labeled.iloc[len(train_raw) :]
        common_required = [*control_features, *M1_PRICE_FEATURES, *SPREAD_FEATURES]
        train = train_labeled[train_labeled["target"].isin([0, 1, 2])].dropna(subset=common_required).copy()
        validation = validation_labeled[validation_labeled["target"].isin([0, 1, 2])].dropna(subset=common_required).copy()
        if set(train["target"].unique()) != {0, 1, 2} or set(validation["target"].unique()) != {0, 1, 2}:
            raise RuntimeError(f"Horizon {horizon} lost a target class")
        distributions[str(horizon)] = {
            "horizon_minutes": horizon * 15,
            "train": _distribution(train),
            "validation": _distribution(validation),
            "same_bar_ties_excluded": int((target_rows["first_passage_target"] == TIE_LABEL).sum()),
            "validation_tail_excluded": horizon,
        }
        retained_frames[str(horizon)] = validation
        m15_name = f"m15_h{horizon}"
        m1_name = f"m1_h{horizon}"
        feature_groups[m15_name] = control_features
        feature_groups[m1_name] = [*control_features, *M1_PRICE_FEATURES]
        if horizon == 5:
            results[m15_name] = {"validation": evaluate(frozen_control, validation), "frozen": True}
            frozen_folds = pd.DataFrame(first_passage_report["cv"]["folds"])
            cv_frames.append(
                pd.DataFrame(
                    {
                        "treatment": m15_name,
                        "fold": frozen_folds["fold"],
                        "direction_auc": frozen_folds["direction_roc_auc"],
                        "balanced_accuracy": frozen_folds["balanced_accuracy"],
                        "buy_recall": frozen_folds["buy_recall"],
                        "sell_recall": frozen_folds["sell_recall"],
                        "directional_macro_f1": frozen_folds["macro_f1"],
                        "activity_auc": frozen_folds["filter_roc_auc"],
                        "scaler_fit_rows": frozen_folds["scaler_fit_rows"],
                        "source": "frozen_first_passage_report",
                    }
                )
            )
        else:
            model, evaluation, folds = fit_and_cv(
                m15_name, train, validation, feature_groups[m15_name], parameters, output_dir
            )
            model.save(f"{EXPERIMENT_ID}_{m15_name}")
            models[m15_name], results[m15_name] = model, evaluation
            cv_frames.append(folds)
        model, evaluation, folds = fit_and_cv(
            m1_name, train, validation, feature_groups[m1_name], parameters, output_dir
        )
        model.save(f"{EXPERIMENT_ID}_{m1_name}")
        models[m1_name], results[m1_name] = model, evaluation
        cv_frames.append(folds)
        if horizon == 5:
            for name, features in {
                "spread_h5": [*control_features, *SPREAD_FEATURES],
                "m1_spread_h5": [*control_features, *M1_PRICE_FEATURES, *SPREAD_FEATURES],
            }.items():
                feature_groups[name] = features
                model, evaluation, folds = fit_and_cv(
                    name, train, validation, features, parameters, output_dir
                )
                model.save(f"{EXPERIMENT_ID}_{name}")
                models[name], results[name] = model, evaluation
                cv_frames.append(folds)
    cv = pd.concat(cv_frames, ignore_index=True)
    comparison_rows = []
    for name, result in results.items():
        validation_metrics = result["validation"]["directional"]
        fold_auc = cv.loc[cv["treatment"] == name, "direction_auc"].astype(float)
        comparison_rows.append(
            {
                "treatment": name,
                "horizon_minutes": int(name.split("h")[-1]) * 15 if "_h" in name else 75,
                **{key: validation_metrics[key] for key in (
                    "roc_auc", "balanced_accuracy", "buy_recall", "sell_recall",
                    "buy_precision", "sell_precision", "directional_macro_f1", "directional_rows"
                )},
                "activity_auc": result["validation"]["activity_auc"],
                "cv_direction_auc_mean": float(fold_auc.mean()),
                "cv_direction_auc_std": float(fold_auc.std(ddof=0)),
                "classification": classify_candidate(
                    float(validation_metrics["roc_auc"]), float(fold_auc.mean()), float(fold_auc.std(ddof=0))
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
    best = comparison.iloc[0].to_dict()
    verdict = str(best["classification"])
    for name, model in models.items():
        validation = retained_frames[name.split("_h")[-1]] if "_h" in name else retained_frames["5"]
        values = validation[feature_groups[name]].to_numpy()
        original = model.model.predict(model.scaler.transform(values))
        reloaded = XGBoostModel(models_dir=str(output_dir))
        reloaded.load(f"{EXPERIMENT_ID}_{name}")
        restored = reloaded.model.predict(reloaded.scaler.transform(values))
        if not np.array_equal(original, restored) or reloaded.feature_columns != feature_groups[name]:
            raise RuntimeError(f"Deterministic reload failed for {name}")
    hashes_after = protected_hashes()
    experiments_after = {
        relative: sha256(BASELINE_DIR / relative) for relative in experiments_before
    }
    if hashes_before != hashes_after or experiments_before != experiments_after:
        raise RuntimeError("Protected artifact changed")
    if verdict in {"PROMISING_DIRECTIONAL_SIGNAL", "ROBUST_DIRECTIONAL_SIGNAL"}:
        best_name = str(best["treatment"])
        best_horizon = int(best["horizon_minutes"]) // 15
        best_model = frozen_control if best_name == "m15_h5" else models[best_name]
        economic = economic_sanity_check(
            best_model,
            retained_frames[str(best_horizon)],
            feature_groups[best_name],
            ohlc,
            best_horizon,
        )
    else:
        economic = {
            "performed": False,
            "reason": "No treatment met the predeclared promising or robust threshold.",
            "historical_spread_available": True,
            "historical_slippage_available": False,
            "after_cost_result": "UNAVAILABLE_RELIABLE_HISTORY",
        }
    feature_manifest = [
        {
            "feature": feature,
            "family": "m1_price_or_tick_volume",
            "maximum_source_timestamp_field": "maximum_source_timestamp",
            "causality_rule": "completed M1 bar close <= prediction_timestamp",
        }
        for feature in M1_PRICE_FEATURES
    ] + [
        {
            "feature": feature,
            "family": "historical_m1_spread",
            "maximum_source_timestamp_field": "maximum_source_timestamp",
            "causality_rule": "completed M1 bar close <= prediction_timestamp",
        }
        for feature in SPREAD_FEATURES
    ]
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_question": "Can M1 price structure or a different ATR-scaled first-passage horizon recover BUY-vs-SELL information?",
        "predeclared_thresholds": DECISION_THRESHOLDS,
        "data_availability": {
            "m1": "AVAILABLE_RELIABLE_HISTORY",
            "m1_manifest": m1_manifest,
            "manifest_verification": manifest_verification,
            "historical_ticks": "UNAVAILABLE_RELIABLE_HISTORY",
            "bid_ask_imbalance": "UNAVAILABLE_RELIABLE_HISTORY",
            "real_volume_usable": bool(m1_manifest["m1"]["historical_coverage"]["nonzero_real_volume_rows"]),
            "historical_spread": "AVAILABLE_IN_M1_RATES",
            "slippage": "UNAVAILABLE_RELIABLE_HISTORY",
        },
        "target": {
            "type": "first passage",
            "barrier": "+/- 1 ATR[t]",
            "horizons_bars": list(HORIZONS),
            "horizons_minutes": [value * 15 for value in HORIZONS],
            "same_bar_ties": "excluded",
            "unresolved": "NO_TRADE",
        },
        "feature_manifest": feature_manifest,
        "feature_groups": {name: features for name, features in feature_groups.items()},
        "timestamp_integrity": {
            "prediction_semantics": "M15 bar close (bar open + 15 minutes)",
            "m1_availability": "M1 bar open + 1 minute",
            "audited_rows": int(timestamp_audit["maximum_source_timestamp"].notna().sum()),
            "future_source_rows": 0,
            "maximum_source_age_minutes": float(
                ((timestamp_audit.index - timestamp_audit["maximum_source_timestamp"]).dt.total_seconds() / 60).max()
            ),
        },
        "data_access": access,
        "distributions": distributions,
        "model_parameters": {**parameters, "objective": "multi:softprob", "eval_metric": "mlogloss", "hyperparameter_tuning": False},
        "comparison": comparison.to_dict(orient="records"),
        "cv_folds": cv.astype(object).where(pd.notna(cv), None).to_dict(orient="records"),
        "decision": {
            "verdict": verdict,
            "best_treatment": best["treatment"],
            "validation_auc": best["roc_auc"],
            "cv_auc_mean": best["cv_direction_auc_mean"],
            "cv_auc_std": best["cv_direction_auc_std"],
        },
        "economic_sanity_check": economic,
        "integrity": {
            "protected_hashes_before": hashes_before,
            "protected_hashes_after": hashes_after,
            "protected_artifacts_checked": len(experiments_before),
            "protected_artifacts_unchanged": True,
            "test_rows_exposed": 0,
            "test_labels_read": False,
            "test_features_read": False,
            "reload_predictions_identical": True,
            "feature_orders_preserved": True,
            "scaler_fit_counts_verified": True,
        },
    }
    comparison.to_csv(output_dir / "validation_comparison.csv", index=False)
    cv.to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(feature_manifest).to_csv(output_dir / "feature_manifest.csv", index=False)
    timestamp_audit.to_parquet(output_dir / "feature_source_timestamps.parquet")
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        f"# {EXPERIMENT_ID}", "", "## Decision",
        f"**{verdict}**: best={best['treatment']}, validation AUC={best['roc_auc']:.4f}, "
        f"CV AUC={best['cv_direction_auc_mean']:.4f} +/- {best['cv_direction_auc_std']:.4f}.",
        "", "## Validation Results", "| Treatment | Horizon | AUC | CV mean | CV std | Classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.treatment} | {row.horizon_minutes}m | {row.roc_auc:.4f} | "
            f"{row.cv_direction_auc_mean:.4f} | {row.cv_direction_auc_std:.4f} | {row.classification} |"
        )
    lines.extend([
        "", "## Availability",
        "- M1 OHLC, tick volume, and historical bar spread: available with verified MT5 provenance.",
        "- Historical ticks and bid/ask imbalance: UNAVAILABLE_RELIABLE_HISTORY.",
        "- Real volume is included in provenance and used only if nonzero; it is not a model feature.",
    ])
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment()["decision"], indent=2))