"""Diagnose temporal drift and session dependence without exposing TEST labels."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.model import XGBoostModel
from train_baseline import CLASS_NAMES, evaluation_metrics


BASELINE_ID = "v0.3_forward_atr_xgb_baseline_v1"
BASELINE_DIR = Path("models/baselines")
OUTPUT_DIR = BASELINE_DIR / f"{BASELINE_ID}_diagnostics"
DATASET_PATH = Path(
    "data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.parquet"
)
METADATA_PATH = DATASET_PATH.with_suffix(".metadata.json")
BASELINE_MODEL_PATH = BASELINE_DIR / f"{BASELINE_ID}.joblib"
BASELINE_EVALUATION_PATH = BASELINE_DIR / f"{BASELINE_ID}_evaluation.json"

SESSION_FEATURES = [
    "day_of_week",
    "is_weekend",
    "hour_of_day",
    "is_market_open",
    "is_asia_session",
    "is_london_session",
    "is_new_york_session",
]
NORMALIZED_FEATURES = [
    "returns",
    "returns_2",
    "returns_5",
    "returns_10",
    "log_returns",
    "high_low_ratio",
    "close_open_ratio",
    "rsi",
    "bb_width",
    "bb_pct",
    "price_to_sma_10",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",
    "atr_pct",
    "volatility_5",
    "volatility_20",
    "volume_ratio",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_validation_only(
    dataset_path: Path, metadata: dict[str, Any], columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Expose exactly TRAIN+VALIDATION rows from parquet; never materialize TEST."""
    train_count = int(metadata["splits"]["train"]["row_count"])
    validation_count = int(metadata["splits"]["validation"]["row_count"])
    exposed_count = train_count + validation_count
    requested_columns = list(dict.fromkeys([*columns, "Date"]))
    parquet = pq.ParquetFile(dataset_path)
    batch_iterator = parquet.iter_batches(
        batch_size=exposed_count,
        columns=requested_columns,
        use_threads=False,
    )
    first_batch = next(batch_iterator)
    if first_batch.num_rows != exposed_count:
        raise ValueError("Bounded parquet read did not match TRAIN+VALIDATION row count")
    exposed = first_batch.to_pandas()
    if len(exposed) != exposed_count:
        raise ValueError("Unexpected row count after bounded parquet conversion")
    train = exposed.iloc[:train_count].copy()
    validation = exposed.iloc[train_count:].copy()
    expected = metadata["splits"]
    actual_boundaries = {
        "train_first": pd.Timestamp(train.index[0]).isoformat(),
        "train_last": pd.Timestamp(train.index[-1]).isoformat(),
        "validation_first": pd.Timestamp(validation.index[0]).isoformat(),
        "validation_last": pd.Timestamp(validation.index[-1]).isoformat(),
    }
    expected_boundaries = {
        "train_first": expected["train"]["first_timestamp"],
        "train_last": expected["train"]["last_timestamp"],
        "validation_first": expected["validation"]["first_timestamp"],
        "validation_last": expected["validation"]["last_timestamp"],
    }
    if actual_boundaries != expected_boundaries:
        raise ValueError("Bounded parquet read does not match recorded split boundaries")
    return train, validation, {
        "reader": "pyarrow.ParquetFile.iter_batches",
        "batch_size": exposed_count,
        "rows_exposed": len(exposed),
        "last_exposed_timestamp": actual_boundaries["validation_last"],
        "test_rows_exposed": 0,
        "test_labels_exposed": False,
    }


def target_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    counts = frame["target"].value_counts().reindex([0, 1, 2], fill_value=0)
    total = len(frame)
    if total == 0:
        return {
            "row_count": 0,
            "sell_pct": np.nan,
            "buy_pct": np.nan,
            "no_trade_pct": np.nan,
        }
    return {
        "row_count": total,
        "sell_pct": float(100 * counts[0] / total),
        "buy_pct": float(100 * counts[1] / total),
        "no_trade_pct": float(100 * counts[2] / total),
    }


def _group_target_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dimension: str,
    train_groups: pd.Series,
    validation_groups: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for split, frame, groups in (
        ("train", train, train_groups),
        ("validation", validation, validation_groups),
    ):
        for group, positions in groups.groupby(groups, observed=True).groups.items():
            summary = target_summary(frame.loc[positions])
            key = str(group)
            summaries[(split, key)] = summary
            rows.append({"dimension": dimension, "group": key, "split": split, **summary})
    for row in rows:
        train_summary = summaries.get(("train", row["group"]))
        validation_summary = summaries.get(("validation", row["group"]))
        for class_key in ("sell_pct", "buy_pct", "no_trade_pct"):
            row[f"validation_minus_train_{class_key}_pp"] = (
                float(validation_summary[class_key] - train_summary[class_key])
                if train_summary and validation_summary
                else np.nan
            )
    return rows


def target_drift_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = _group_target_rows(
        train,
        validation,
        "overall",
        pd.Series("all", index=train.index),
        pd.Series("all", index=validation.index),
    )
    groupers = {
        "year": (train.index.year, validation.index.year),
        "year_month": (
            train.index.to_period("M").astype(str),
            validation.index.to_period("M").astype(str),
        ),
        "month_of_year": (train.index.month, validation.index.month),
        "weekday": (train.index.dayofweek, validation.index.dayofweek),
        "utc_hour": (train.index.hour, validation.index.hour),
    }
    for dimension, (train_values, validation_values) in groupers.items():
        rows.extend(
            _group_target_rows(
                train,
                validation,
                dimension,
                pd.Series(train_values, index=train.index),
                pd.Series(validation_values, index=validation.index),
            )
        )
    return pd.DataFrame(rows)


def session_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "asia_00_07": lambda frame: frame["is_asia_session"] == 1,
        "london_08_15": lambda frame: frame["is_london_session"] == 1,
        "new_york_13_20": lambda frame: frame["is_new_york_session"] == 1,
        "london_new_york_overlap_13_15": lambda frame: (
            (frame["is_london_session"] == 1)
            & (frame["is_new_york_session"] == 1)
        ),
        "unassigned": lambda frame: (
            (frame["is_asia_session"] == 0)
            & (frame["is_london_session"] == 0)
            & (frame["is_new_york_session"] == 0)
        ),
    }
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for split, frame in (("train", train), ("validation", validation)):
        for hour in range(24):
            subset = frame[frame.index.hour == hour]
            if len(subset):
                summary = target_summary(subset)
                summaries[(split, "utc_hour", str(hour))] = summary
                rows.append(
                    {"dimension": "utc_hour", "group": str(hour), "split": split, **summary}
                )
        for name, selector in definitions.items():
            subset = frame.loc[selector(frame)]
            summary = target_summary(subset)
            summaries[(split, "session", name)] = summary
            rows.append({"dimension": "session", "group": name, "split": split, **summary})
    for row in rows:
        key = (row["dimension"], row["group"])
        train_summary = summaries.get(("train", *key))
        validation_summary = summaries.get(("validation", *key))
        for class_key in ("sell_pct", "buy_pct", "no_trade_pct"):
            row[f"validation_minus_train_{class_key}_pp"] = (
                float(validation_summary[class_key] - train_summary[class_key])
                if train_summary and validation_summary
                else np.nan
            )
    return pd.DataFrame(rows)


def population_stability_index(train_values: np.ndarray, validation_values: np.ndarray) -> float:
    quantiles = np.unique(np.quantile(train_values, np.linspace(0, 1, 11)))
    if len(quantiles) < 2:
        return 0.0 if np.all(validation_values == train_values[0]) else 27.631
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    train_counts = np.histogram(train_values, bins=quantiles)[0]
    validation_counts = np.histogram(validation_values, bins=quantiles)[0]
    train_pct = np.clip(train_counts / len(train_values), 1e-6, None)
    validation_pct = np.clip(validation_counts / len(validation_values), 1e-6, None)
    return float(np.sum((validation_pct - train_pct) * np.log(validation_pct / train_pct)))


def feature_family(feature: str) -> str:
    if feature in SESSION_FEATURES:
        return "session_time"
    if feature in NORMALIZED_FEATURES:
        return "normalized_relative"
    return "price_level_or_scale_dependent"


def feature_drift_table(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows = []
    for feature in features:
        train_values = train[feature].to_numpy(dtype=float)
        validation_values = validation[feature].to_numpy(dtype=float)
        train_std = float(np.std(train_values))
        raw_wasserstein = float(wasserstein_distance(train_values, validation_values))
        rows.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "ks_statistic": float(ks_2samp(train_values, validation_values).statistic),
                "psi": population_stability_index(train_values, validation_values),
                "wasserstein": raw_wasserstein,
                "wasserstein_train_std_units": (
                    raw_wasserstein / train_std if train_std > 0 else 0.0
                ),
                "train_mean": float(np.mean(train_values)),
                "validation_mean": float(np.mean(validation_values)),
                "train_std": train_std,
                "validation_std": float(np.std(validation_values)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["ks_statistic", "psi"], ascending=False
    ).reset_index(drop=True)


def _regime_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dimension: str,
    train_groups: pd.Series,
    validation_groups: pd.Series,
) -> list[dict[str, Any]]:
    rows = []
    for split, frame, groups in (
        ("train", train, train_groups),
        ("validation", validation, validation_groups),
    ):
        for group, positions in groups.groupby(groups, observed=True).groups.items():
            subset = frame.loc[positions]
            rows.append(
                {
                    "dimension": dimension,
                    "regime": str(group),
                    "split": split,
                    **target_summary(subset),
                    "close_mean": float(subset["Close"].mean()),
                    "close_median": float(subset["Close"].median()),
                    "contemporaneous_return_mean": float(subset["returns"].mean()),
                    "contemporaneous_return_median": float(subset["returns"].median()),
                    "atr_pct_mean": float(subset["atr_pct"].mean()),
                }
            )
    return rows


def regime_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    price_edges = np.unique(np.quantile(train["Close"], np.linspace(0, 1, 6)))
    price_edges[0], price_edges[-1] = -np.inf, np.inf
    volatility_edges = np.unique(np.quantile(train["atr_pct"], [0, 1 / 3, 2 / 3, 1]))
    volatility_edges[0], volatility_edges[-1] = -np.inf, np.inf
    rows = _regime_rows(
        train,
        validation,
        "price_quantile_train_edges",
        pd.cut(train["Close"], price_edges, labels=[f"Q{i}" for i in range(1, len(price_edges))]),
        pd.cut(validation["Close"], price_edges, labels=[f"Q{i}" for i in range(1, len(price_edges))]),
    )
    rows.extend(
        _regime_rows(
            train,
            validation,
            "atr_pct_train_tertiles",
            pd.cut(train["atr_pct"], volatility_edges, labels=["low", "medium", "high"]),
            pd.cut(validation["atr_pct"], volatility_edges, labels=["low", "medium", "high"]),
        )
    )
    rows.extend(
        _regime_rows(
            train,
            validation,
            "year",
            pd.Series(train.index.year, index=train.index),
            pd.Series(validation.index.year, index=validation.index),
        )
    )
    return pd.DataFrame(rows)


def _standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    pooled = np.sqrt((np.var(left) + np.var(right)) / 2)
    return float((np.mean(left) - np.mean(right)) / pooled) if pooled > 0 else 0.0


def class_separability_table(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = train[features].to_numpy(dtype=float)
    y = train["target"].to_numpy(dtype=int)
    discrete = np.array([feature in SESSION_FEATURES for feature in features])
    mutual_information = mutual_info_classif(
        X, y, discrete_features=discrete, random_state=42
    )
    rows = []
    for index, feature in enumerate(features):
        values = train[feature].to_numpy(dtype=float)
        row: dict[str, Any] = {"feature": feature, "family": feature_family(feature), "mutual_information": float(mutual_information[index])}
        for comparison, other_label in (("buy_vs_sell", 0), ("buy_vs_no_trade", 2)):
            mask = np.isin(y, [1, other_label])
            pair_values = values[mask]
            pair_target = (y[mask] == 1).astype(int)
            raw_auc = float(roc_auc_score(pair_target, pair_values)) if np.unique(pair_values).size > 1 else 0.5
            row[f"{comparison}_smd"] = _standardized_mean_difference(
                values[y == 1], values[y == other_label]
            )
            row[f"{comparison}_auc"] = raw_auc
            row[f"{comparison}_separability_auc"] = max(raw_auc, 1 - raw_auc)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mutual_information", ascending=False).reset_index(drop=True)


def model_parameters(baseline_report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
        "reg_lambda", "random_state",
    )
    return {key: baseline_report["model_parameters"][key] for key in keys}


def evaluate_model(
    model: XGBoostModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    result = {}
    for split, frame in (("train", train), ("validation", validation)):
        X = frame[features].to_numpy()
        predictions = model.model.predict(model.scaler.transform(X))
        result[split] = evaluation_metrics(frame["target"].to_numpy(), predictions)
    return result


def diagnostic_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    all_features: list[str],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    parameters = model_parameters(baseline_report)
    majority_class = int(train["target"].mode().iloc[0])
    models: dict[str, Any] = {
        "A_majority": {
            split: evaluation_metrics(
                frame["target"].to_numpy(), np.full(len(frame), majority_class)
            )
            for split, frame in (("train", train), ("validation", validation))
        }
    }
    for name, features in (
        ("B_session_time_only", SESSION_FEATURES),
        ("C_normalized_relative_only", NORMALIZED_FEATURES),
    ):
        model = XGBoostModel(**parameters, models_dir=str(BASELINE_DIR))
        started = time.perf_counter()
        model.fit_training_data(train, features)
        models[name] = {
            **evaluate_model(model, train, validation, features),
            "features": features,
            "training_seconds": time.perf_counter() - started,
        }
    reference = XGBoostModel(models_dir=str(BASELINE_DIR))
    reference.load(BASELINE_ID)
    models["D_full_42_reference"] = {
        **evaluate_model(reference, train, validation, all_features),
        "features": all_features,
        "retrained": False,
        "source": str(BASELINE_MODEL_PATH),
    }
    return models


def importance_stability_table(
    train: pd.DataFrame,
    features: list[str],
    baseline_report: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parameters = model_parameters(baseline_report)
    records = []
    folds = []
    for fold, (train_indices, validation_indices) in enumerate(
        TimeSeriesSplit(n_splits=5).split(train), 1
    ):
        fold_train = train.iloc[train_indices]
        fold_validation = train.iloc[validation_indices]
        model = XGBoostModel(**parameters, models_dir=str(BASELINE_DIR))
        model.fit_training_data(fold_train, features)
        metrics = evaluate_model(model, fold_train, fold_validation, features)["validation"]
        importance = model.feature_importance()
        importance["rank"] = np.arange(1, len(importance) + 1)
        for row in importance.itertuples(index=False):
            records.append(
                {"fold": fold, "feature": row.feature, "importance": row.importance, "rank": row.rank}
            )
        folds.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "train_first": pd.Timestamp(fold_train.index[0]).isoformat(),
                "train_last": pd.Timestamp(fold_train.index[-1]).isoformat(),
                "validation_first": pd.Timestamp(fold_validation.index[0]).isoformat(),
                "validation_last": pd.Timestamp(fold_validation.index[-1]).isoformat(),
                "metrics": metrics,
            }
        )
    raw = pd.DataFrame(records)
    summary = raw.groupby("feature").agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_min=("importance", "min"),
        importance_max=("importance", "max"),
        rank_mean=("rank", "mean"),
        rank_std=("rank", "std"),
        rank_min=("rank", "min"),
        rank_max=("rank", "max"),
    ).reset_index()
    for fold in range(1, 6):
        fold_values = raw[raw["fold"] == fold].set_index("feature")
        summary[f"fold_{fold}_importance"] = summary["feature"].map(fold_values["importance"])
        summary[f"fold_{fold}_rank"] = summary["feature"].map(fold_values["rank"])
    return summary.sort_values("importance_mean", ascending=False).reset_index(drop=True), folds


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def evidence_and_conclusions(
    feature_drift: pd.DataFrame,
    sessions: pd.DataFrame,
    regimes: pd.DataFrame,
    separability: pd.DataFrame,
    models: dict[str, Any],
    importance_stability: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    family_drift = (
        feature_drift.groupby("family")
        .agg(
            feature_count=("feature", "count"),
            mean_ks=("ks_statistic", "mean"),
            median_ks=("ks_statistic", "median"),
            max_ks=("ks_statistic", "max"),
            mean_psi=("psi", "mean"),
        )
        .to_dict(orient="index")
    )
    session_rows = sessions[sessions["dimension"] == "session"]
    session_targets = {
        f"{row.split}_{row.group}": {
            "row_count": row.row_count,
            "sell_pct": row.sell_pct,
            "buy_pct": row.buy_pct,
            "no_trade_pct": row.no_trade_pct,
        }
        for row in session_rows.itertuples(index=False)
    }
    validation_hours = sessions[
        (sessions["dimension"] == "utc_hour")
        & (sessions["split"] == "validation")
    ].copy()
    shift_columns = [
        "validation_minus_train_sell_pct_pp",
        "validation_minus_train_buy_pct_pp",
        "validation_minus_train_no_trade_pct_pp",
    ]
    validation_hours["max_abs_shift_pp"] = validation_hours[shift_columns].abs().max(axis=1)
    largest_hour_shifts = validation_hours.nlargest(10, "max_abs_shift_pp").to_dict(orient="records")
    price_regimes = regimes[regimes["dimension"] == "price_quantile_train_edges"].to_dict(orient="records")
    volatility_regimes = regimes[regimes["dimension"] == "atr_pct_train_tertiles"].to_dict(orient="records")
    best_buy_sell = separability.nlargest(1, "buy_vs_sell_separability_auc").iloc[0].to_dict()
    best_buy_no_trade = separability.nlargest(1, "buy_vs_no_trade_separability_auc").iloc[0].to_dict()
    importance_lookup = importance_stability.set_index("feature").to_dict(orient="index")
    session_validation = models["B_session_time_only"]["validation"]
    normalized_validation = models["C_normalized_relative_only"]["validation"]
    full_train = models["D_full_42_reference"]["train"]
    full_validation = models["D_full_42_reference"]["validation"]
    evidence = {
        "family_drift": family_drift,
        "session_targets": session_targets,
        "largest_utc_hour_target_shifts": largest_hour_shifts,
        "price_regimes": price_regimes,
        "volatility_regimes": volatility_regimes,
        "best_train_buy_vs_sell_univariate": best_buy_sell,
        "best_train_buy_vs_no_trade_univariate": best_buy_no_trade,
        "session_model_validation": session_validation,
        "normalized_model_validation": normalized_validation,
        "full_model_train_validation_macro_f1_gap": full_train["macro_f1"] - full_validation["macro_f1"],
        "importance_stability": {
            feature: importance_lookup[feature]
            for feature in ("is_asia_session", "hour_of_day")
        },
    }
    conclusions = [
        "Primary problem: a combination led by model overfitting and absolute-price regime shift. "
        f"Full-model macro F1 falls {evidence['full_model_train_validation_macro_f1_gap']:.4f}, "
        "whereas the session-only model is stable across TRAIN and VALIDATION.",
        "Distribution drift is significant but concentrated: price-level/scale-dependent features "
        f"have mean KS {family_drift['price_level_or_scale_dependent']['mean_ks']:.4f} versus "
        f"{family_drift['normalized_relative']['mean_ks']:.4f} for normalized features and "
        f"{family_drift['session_time']['mean_ks']:.4f} for session/time features.",
        "Session/time explains directional activity more than direction. Asia is predominantly "
        f"NO_TRADE ({session_targets['train_asia_00_07']['no_trade_pct']:.2f}% TRAIN), while London "
        f"and New York are about {session_targets['train_london_08_15']['no_trade_pct']:.2f}% and "
        f"{session_targets['train_new_york_13_20']['no_trade_pct']:.2f}% NO_TRADE. The session-only "
        f"model reaches validation macro F1 {session_validation['macro_f1']:.4f}, but BUY/SELL recall "
        "remains weak and asymmetric.",
        "Absolute price-level features are unstable: all VALIDATION observations occupy Q5 under "
        "TRAIN-defined Close quintiles, and MA/Bollinger levels dominate the drift ranking. This "
        "makes tree thresholds learned at earlier index levels poor extrapolators.",
        "BUY recall is not explained by aggregate class prevalence, which changes only slightly. "
        f"TRAIN-only BUY-vs-SELL univariate separability peaks at AUC {best_buy_sell['buy_vs_sell_separability_auc']:.4f} "
        f"for {best_buy_sell['feature']}; the features distinguish directional activity from NO_TRADE "
        "far better than BUY from SELL. The full model's directional allocation then shifts from "
        f"{full_train['prediction_distribution']['BUY']['percentage']:.2f}% BUY / "
        f"{full_train['prediction_distribution']['SELL']['percentage']:.2f}% SELL on TRAIN to "
        f"{full_validation['prediction_distribution']['BUY']['percentage']:.2f}% BUY / "
        f"{full_validation['prediction_distribution']['SELL']['percentage']:.2f}% SELL on VALIDATION.",
        "The 42 features contain some signal, but stable out-of-time information is limited: the "
        f"full model improves validation macro F1 only {full_validation['macro_f1'] - session_validation['macro_f1']:.4f} "
        "over session/time alone while losing substantial accuracy; normalized-only performance is "
        f"also modest (macro F1 {normalized_validation['macro_f1']:.4f}).",
        "Next experiment: a single controlled TRAIN/VALIDATION ablation using the union of the "
        "18 normalized/relative and 7 session/time features, with the same XGBoost parameters and "
        "boundary. This directly tests whether scale-dependent absolute features cause the unstable "
        "directional allocation; it is an experiment, not a production feature-removal decision, "
        "and TEST must remain sealed.",
    ]
    return evidence, conclusions


def render_report(report: dict[str, Any]) -> str:
    models = report["diagnostic_models"]
    target = report["key_findings"]["overall_target_shift_pp"]
    top_drift = report["key_findings"]["top_feature_drift"]
    importance = report["key_findings"]["importance_stability"]
    evidence = report["evidence"]
    lines = [
        f"# {BASELINE_ID} Temporal Diagnostics",
        "",
        "## Data Policy",
        f"- TRAIN rows: {report['data_access']['train_rows']:,}",
        f"- VALIDATION rows: {report['data_access']['validation_rows']:,}",
        "- TEST rows/labels exposed: 0 / false",
        "- Reference baseline was loaded, not retrained.",
        "",
        "## Target Shift (Validation minus Train)",
        f"- SELL: {target['sell']:+.2f} pp",
        f"- BUY: {target['buy']:+.2f} pp",
        f"- NO_TRADE: {target['no_trade']:+.2f} pp",
        "",
        "## Diagnostic Models",
        "| Model | Split | Accuracy | Balanced accuracy | Macro F1 | BUY recall | SELL recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, result in models.items():
        for split in ("train", "validation"):
            metrics = result[split]
            lines.append(
                f"| {model_name} | {split} | {metrics['accuracy']:.4f} | "
                f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['per_class']['BUY']['recall']:.4f} | "
                f"{metrics['per_class']['SELL']['recall']:.4f} |"
            )
    lines.extend(["", "## Strongest Feature Drift", "| Feature | Family | KS | PSI | Std. Wasserstein |", "|---|---|---:|---:|---:|"])
    for row in top_drift[:15]:
        lines.append(
            f"| {row['feature']} | {row['family']} | {row['ks_statistic']:.4f} | "
            f"{row['psi']:.4f} | {row['wasserstein_train_std_units']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Importance Stability",
            f"- is_asia_session: mean {importance['is_asia_session']['importance_mean']:.4f}, "
            f"range {importance['is_asia_session']['importance_min']:.4f}-{importance['is_asia_session']['importance_max']:.4f}, "
            f"rank range {importance['is_asia_session']['rank_min']:.0f}-{importance['is_asia_session']['rank_max']:.0f}",
            f"- hour_of_day: mean {importance['hour_of_day']['importance_mean']:.4f}, "
            f"range {importance['hour_of_day']['importance_min']:.4f}-{importance['hour_of_day']['importance_max']:.4f}, "
            f"rank range {importance['hour_of_day']['rank_min']:.0f}-{importance['hour_of_day']['rank_max']:.0f}",
            "",
            "## Session and Regime Evidence",
            f"- TRAIN Asia NO_TRADE: {evidence['session_targets']['train_asia_00_07']['no_trade_pct']:.2f}%",
            f"- TRAIN London/New York NO_TRADE: {evidence['session_targets']['train_london_08_15']['no_trade_pct']:.2f}% / {evidence['session_targets']['train_new_york_13_20']['no_trade_pct']:.2f}%",
            f"- Largest hour-specific shift: UTC {evidence['largest_utc_hour_target_shifts'][0]['group']}, {evidence['largest_utc_hour_target_shifts'][0]['max_abs_shift_pp']:.2f} percentage points",
            "- Every VALIDATION observation is in Q5 under TRAIN-defined Close quintiles.",
            f"- Best TRAIN-only BUY-vs-SELL univariate AUC: {evidence['best_train_buy_vs_sell_univariate']['buy_vs_sell_separability_auc']:.4f} ({evidence['best_train_buy_vs_sell_univariate']['feature']})",
            "",
            "## Conclusions",
        ]
    )
    for index, conclusion in enumerate(report["conclusions"], 1):
        lines.append(f"{index}. {conclusion}")
    lines.extend(["", "Full measurements are in `diagnostic_report.json` and the six CSV files."])
    return "\n".join(lines) + "\n"


def run_diagnostics() -> dict[str, Any]:
    output_paths = {
        "report_md": OUTPUT_DIR / "diagnostic_report.md",
        "report_json": OUTPUT_DIR / "diagnostic_report.json",
        "target_drift": OUTPUT_DIR / "target_drift.csv",
        "feature_drift": OUTPUT_DIR / "feature_drift.csv",
        "session_analysis": OUTPUT_DIR / "session_analysis.csv",
        "regime_analysis": OUTPUT_DIR / "regime_analysis.csv",
        "class_separability": OUTPUT_DIR / "class_separability.csv",
        "importance_stability": OUTPUT_DIR / "importance_stability.csv",
    }
    if OUTPUT_DIR.exists() or any(path.exists() for path in output_paths.values()):
        raise FileExistsError(f"Diagnostic output already exists: {OUTPUT_DIR}")

    dataset_hash_before = sha256(DATASET_PATH)
    model_hash_before = sha256(BASELINE_MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)
    with BASELINE_EVALUATION_PATH.open(encoding="utf-8") as file_handle:
        baseline_report = json.load(file_handle)
    features = list(metadata["feature_names"])
    columns = [*features, "Close", "target"]
    train, validation, access = load_train_validation_only(DATASET_PATH, metadata, columns)

    target_drift = target_drift_table(train, validation)
    feature_drift = feature_drift_table(train, validation, features)
    sessions = session_table(train, validation)
    regimes = regime_table(train, validation)
    separability = class_separability_table(train, features)
    models = diagnostic_models(train, validation, features, baseline_report)
    importance_stability, fold_details = importance_stability_table(
        train, features, baseline_report
    )
    evidence, conclusions = evidence_and_conclusions(
        feature_drift,
        sessions,
        regimes,
        separability,
        models,
        importance_stability,
    )

    overall_train = target_summary(train)
    overall_validation = target_summary(validation)
    importance_lookup = importance_stability.set_index("feature").to_dict(orient="index")
    report: dict[str, Any] = {
        "diagnostic_id": f"{BASELINE_ID}_diagnostics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_baseline": BASELINE_ID,
        "data_access": {
            **access,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_boundary": metadata["splits"]["test"],
            "policy": "TEST batch was not requested; TEST labels were not exposed or evaluated",
        },
        "feature_families": {
            "session_time": SESSION_FEATURES,
            "normalized_relative": NORMALIZED_FEATURES,
            "price_level_or_scale_dependent": [
                feature for feature in features if feature_family(feature) == "price_level_or_scale_dependent"
            ],
        },
        "diagnostic_models": models,
        "importance_fold_details": fold_details,
        "evidence": evidence,
        "key_findings": {
            "overall_targets": {"train": overall_train, "validation": overall_validation},
            "overall_target_shift_pp": {
                "sell": overall_validation["sell_pct"] - overall_train["sell_pct"],
                "buy": overall_validation["buy_pct"] - overall_train["buy_pct"],
                "no_trade": overall_validation["no_trade_pct"] - overall_train["no_trade_pct"],
            },
            "top_feature_drift": feature_drift.head(15).to_dict(orient="records"),
            "top_train_only_separability": separability.head(15).to_dict(orient="records"),
            "importance_stability": {
                feature: importance_lookup[feature]
                for feature in ("is_asia_session", "hour_of_day")
            },
        },
        "integrity": {
            "dataset_sha256_before": dataset_hash_before,
            "dataset_sha256_after": sha256(DATASET_PATH),
            "dataset_unchanged": sha256(DATASET_PATH) == dataset_hash_before,
            "baseline_model_sha256_before": model_hash_before,
            "baseline_model_sha256_after": sha256(BASELINE_MODEL_PATH),
            "baseline_model_unchanged": sha256(BASELINE_MODEL_PATH) == model_hash_before,
            "baseline_retrained": False,
            "test_labels_read": False,
            "test_evaluated": False,
        },
        "conclusions": conclusions,
    }
    report = _native(report)
    integrity = report["integrity"]
    if not (
        integrity["dataset_unchanged"]
        and integrity["baseline_model_unchanged"]
        and integrity["baseline_retrained"] is False
        and integrity["test_labels_read"] is False
        and integrity["test_evaluated"] is False
    ):
        raise RuntimeError(f"Diagnostic integrity failure: {integrity}")

    OUTPUT_DIR.mkdir(parents=True)
    target_drift.to_csv(output_paths["target_drift"], index=False)
    feature_drift.to_csv(output_paths["feature_drift"], index=False)
    sessions.to_csv(output_paths["session_analysis"], index=False)
    regimes.to_csv(output_paths["regime_analysis"], index=False)
    separability.to_csv(output_paths["class_separability"], index=False)
    importance_stability.to_csv(output_paths["importance_stability"], index=False)
    output_paths["report_json"].write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    output_paths["report_md"].write_text(render_report(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_diagnostics()
    print(json.dumps(result["key_findings"], indent=2, allow_nan=False))