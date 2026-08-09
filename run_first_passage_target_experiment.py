"""Run one controlled five-bar first-passage target experiment."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from diagnose_baseline import load_train_validation_only, sha256
from diagnose_directional_signal import evaluate_saved_model
from diagnose_target_construction import class_distribution
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
from src.model import XGBoostModel
EXPERIMENT_ID = f"{BASELINE_ID}_first_passage_1atr_5bar_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
DIRECTIONAL_DIAGNOSTIC_PATH = (
    BASELINE_DIR
    / f"{BASELINE_ID}_directional_signal_diagnostic_v1"
    / "diagnostic_report.json"
)
HORIZON = 5
BARRIER_MULTIPLIER = 1.0
TIE_LABEL = 3
INCOMPLETE_LABEL = 4
OHLC_RECONSTRUCTION_COLUMNS = [
    "atr",
    "atr_pct",
    "close_open_ratio",
    "upper_shadow",
    "lower_shadow",
]
REFERENCE_SPECS = {
    "endpoint_42f": {
        "model": BASELINE_MODEL_PATH,
        "report": BASELINE_REPORT_PATH,
    },
    "endpoint_25f": {
        "model": BASELINE_DIR / f"{BASELINE_ID}_ablation_25f" / f"{BASELINE_ID}_ablation_25f.joblib",
        "report": BASELINE_DIR / f"{BASELINE_ID}_ablation_25f" / "experiment_report.json",
    },
    "endpoint_18f": {
        "model": BASELINE_DIR / f"{BASELINE_ID}_ablation_18f_normalized" / f"{BASELINE_ID}_ablation_18f_normalized.joblib",
        "report": BASELINE_DIR / f"{BASELINE_ID}_ablation_18f_normalized" / "experiment_report.json",
    },
}


def reconstruct_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["atr"] / frame["atr_pct"]
    open_price = close / (1 + frame["close_open_ratio"])
    high = pd.concat([open_price, close], axis=1).max(axis=1) + frame["upper_shadow"]
    low = pd.concat([open_price, close], axis=1).min(axis=1) - frame["lower_shadow"]
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close},
        index=frame.index,
    )


def build_first_passage_target(
    ohlc: pd.DataFrame,
    atr: pd.Series,
    *,
    horizon: int = HORIZON,
    multiplier: float = BARRIER_MULTIPLIER,
) -> pd.DataFrame:
    if not ohlc.index.equals(atr.index):
        raise ValueError("OHLC and ATR indices must match")
    labels = np.full(len(ohlc), INCOMPLETE_LABEL, dtype=int)
    first_hit_bar = np.full(len(ohlc), np.nan)
    upper_barriers = ohlc["Close"].to_numpy() + multiplier * atr.to_numpy()
    lower_barriers = ohlc["Close"].to_numpy() - multiplier * atr.to_numpy()
    highs = ohlc["High"].to_numpy()
    lows = ohlc["Low"].to_numpy()
    complete_count = max(0, len(ohlc) - horizon)
    labels[:complete_count] = 2
    for row in range(complete_count):
        for offset in range(1, horizon + 1):
            upper_hit = highs[row + offset] >= upper_barriers[row]
            lower_hit = lows[row + offset] <= lower_barriers[row]
            if upper_hit and lower_hit:
                labels[row] = TIE_LABEL
                first_hit_bar[row] = offset
                break
            if upper_hit:
                labels[row] = 1
                first_hit_bar[row] = offset
                break
            if lower_hit:
                labels[row] = 0
                first_hit_bar[row] = offset
                break
    return pd.DataFrame(
        {
            "first_passage_target": labels,
            "first_hit_bar": first_hit_bar,
            "upper_barrier": upper_barriers,
            "lower_barrier": lower_barriers,
        },
        index=ohlc.index,
    )


def independently_reconstruct_target(
    ohlc: pd.DataFrame, atr: pd.Series, horizon: int = HORIZON
) -> np.ndarray:
    results = []
    for position, (_, row) in enumerate(ohlc.iterrows()):
        if position + horizon >= len(ohlc):
            results.append(INCOMPLETE_LABEL)
            continue
        upper = row["Close"] + atr.iloc[position]
        lower = row["Close"] - atr.iloc[position]
        outcome = 2
        for future_position in range(position + 1, position + horizon + 1):
            future = ohlc.iloc[future_position]
            upper_hit = future["High"] >= upper
            lower_hit = future["Low"] <= lower
            if upper_hit and lower_hit:
                outcome = TIE_LABEL
                break
            if upper_hit:
                outcome = 1
                break
            if lower_hit:
                outcome = 0
                break
        results.append(outcome)
    return np.asarray(results, dtype=int)


def target_distribution(labels: pd.Series) -> list[dict[str, Any]]:
    names = {0: "SELL", 1: "BUY", 2: "NO_TRADE", 3: "SAME_BAR_TIE", 4: "INCOMPLETE"}
    counts = labels.value_counts().reindex(names, fill_value=0)
    return [
        {
            "class": names[label],
            "count": int(counts[label]),
            "percentage": float(100 * counts[label] / len(labels)),
        }
        for label in names
    ]


def prepare_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_labeled = train.copy()
    validation_labeled = validation.copy()
    train_labeled["target"] = target_rows.loc[train.index, "first_passage_target"]
    validation_labeled["target"] = target_rows.loc[validation.index, "first_passage_target"]
    valid_labels = {0, 1, 2}
    return (
        train_labeled[train_labeled["target"].isin(valid_labels)].copy(),
        validation_labeled[validation_labeled["target"].isin(valid_labels)].copy(),
    )


def evaluate_model(model: XGBoostModel, frame: pd.DataFrame) -> dict[str, Any]:
    return evaluate_saved_model(model, frame)


def chronological_cv(
    train: pd.DataFrame, features: list[str], parameters: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    details = []
    for fold, (train_indices, validation_indices) in enumerate(
        TimeSeriesSplit(n_splits=5).split(train), 1
    ):
        fold_train = train.iloc[train_indices]
        fold_validation = train.iloc[validation_indices]
        model = XGBoostModel(**parameters, models_dir=str(BASELINE_DIR))
        model.fit_training_data(fold_train, features)
        result = evaluate_model(model, fold_validation)
        metrics = result["multiclass"]
        row = {
            "row_type": "fold",
            "fold": fold,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "buy_recall": metrics["per_class"]["BUY"]["recall"],
            "sell_recall": metrics["per_class"]["SELL"]["recall"],
            "direction_roc_auc": result["buy_vs_sell_on_actual_directional"]["roc_auc"],
            "filter_roc_auc": result["directional_filter"]["roc_auc"],
            "scaler_fit_rows": int(model.scaler.n_samples_seen_),
        }
        if row["scaler_fit_rows"] != len(fold_train):
            raise RuntimeError(f"Fold {fold} scaler was not fit on exactly its TRAIN rows")
        rows.append(row)
        details.append(
            {
                **row,
                "train_first": pd.Timestamp(fold_train.index[0]).isoformat(),
                "train_last": pd.Timestamp(fold_train.index[-1]).isoformat(),
                "validation_first": pd.Timestamp(fold_validation.index[0]).isoformat(),
                "validation_last": pd.Timestamp(fold_validation.index[-1]).isoformat(),
            }
        )
    folds = pd.DataFrame(rows)
    metrics = [
        "accuracy", "balanced_accuracy", "macro_f1", "buy_recall", "sell_recall",
        "direction_roc_auc", "filter_roc_auc",
    ]
    summaries = []
    for statistic in ("mean", "std", "min", "max"):
        values = folds[metrics].std(ddof=0) if statistic == "std" else getattr(folds[metrics], statistic)()
        summaries.append({"row_type": statistic, "fold": None, **values.to_dict(), "scaler_fit_rows": None})
    return pd.concat([folds, pd.DataFrame(summaries)], ignore_index=True), details


def comparison_table(
    reference_reports: dict[str, dict[str, Any]],
    directional_report: dict[str, Any],
    result: dict[str, Any],
    cv_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    diagnostic_names = {
        "endpoint_42f": "reference_42f",
        "endpoint_25f": "ablation_25f",
        "endpoint_18f": "normalized_18f",
    }
    for name, report in reference_reports.items():
        metrics = report["evaluation"]["validation"]
        diagnostic = directional_report["saved_model_validation"][diagnostic_names[name]]
        if name == "endpoint_42f":
            cv_folds = report["chronological_cv"]["folds"]
            cv_macro_f1 = None
        else:
            cv_folds = report["cv"]["folds"]
            cv_macro_f1 = float(np.mean([fold["macro_f1"] for fold in cv_folds]))
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "sell_precision": metrics["per_class"]["SELL"]["precision"],
                "sell_recall": metrics["per_class"]["SELL"]["recall"],
                "sell_f1": metrics["per_class"]["SELL"]["f1"],
                "buy_precision": metrics["per_class"]["BUY"]["precision"],
                "buy_recall": metrics["per_class"]["BUY"]["recall"],
                "buy_f1": metrics["per_class"]["BUY"]["f1"],
                "no_trade_precision": metrics["per_class"]["NO_TRADE"]["precision"],
                "no_trade_recall": metrics["per_class"]["NO_TRADE"]["recall"],
                "no_trade_f1": metrics["per_class"]["NO_TRADE"]["f1"],
                "sell_prediction_pct": metrics["prediction_distribution"]["SELL"]["percentage"],
                "buy_prediction_pct": metrics["prediction_distribution"]["BUY"]["percentage"],
                "no_trade_prediction_pct": metrics["prediction_distribution"]["NO_TRADE"]["percentage"],
                "direction_roc_auc": diagnostic["buy_vs_sell_on_actual_directional"]["roc_auc"],
                "filter_roc_auc": diagnostic["directional_filter"]["roc_auc"],
                "cv_accuracy_mean": float(np.mean([fold["accuracy"] for fold in cv_folds])),
                "cv_macro_f1_mean": cv_macro_f1,
            }
        )
    metrics = result["multiclass"]
    rows.append(
        {
            "model": "first_passage_42f",
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "sell_precision": metrics["per_class"]["SELL"]["precision"],
            "sell_recall": metrics["per_class"]["SELL"]["recall"],
            "sell_f1": metrics["per_class"]["SELL"]["f1"],
            "buy_precision": metrics["per_class"]["BUY"]["precision"],
            "buy_recall": metrics["per_class"]["BUY"]["recall"],
            "buy_f1": metrics["per_class"]["BUY"]["f1"],
            "no_trade_precision": metrics["per_class"]["NO_TRADE"]["precision"],
            "no_trade_recall": metrics["per_class"]["NO_TRADE"]["recall"],
            "no_trade_f1": metrics["per_class"]["NO_TRADE"]["f1"],
            "sell_prediction_pct": metrics["prediction_distribution"]["SELL"]["percentage"],
            "buy_prediction_pct": metrics["prediction_distribution"]["BUY"]["percentage"],
            "no_trade_prediction_pct": metrics["prediction_distribution"]["NO_TRADE"]["percentage"],
            "direction_roc_auc": result["buy_vs_sell_on_actual_directional"]["roc_auc"],
            "filter_roc_auc": result["directional_filter"]["roc_auc"],
            "cv_accuracy_mean": float(cv_results.loc[cv_results["row_type"] == "mean", "accuracy"].iloc[0]),
            "cv_macro_f1_mean": float(cv_results.loc[cv_results["row_type"] == "mean", "macro_f1"].iloc[0]),
        }
    )
    return pd.DataFrame(rows)


def verdict(comparison: pd.DataFrame) -> dict[str, str]:
    first_passage = comparison.set_index("model").loc["first_passage_42f"]
    endpoint = comparison[comparison["model"] != "first_passage_42f"]
    directional_auc = float(first_passage["direction_roc_auc"])
    filter_auc = float(first_passage["filter_roc_auc"])
    minimum_recall = min(float(first_passage["buy_recall"]), float(first_passage["sell_recall"]))
    best_endpoint_minimum_recall = max(
        min(float(row.buy_recall), float(row.sell_recall))
        for row in endpoint.itertuples(index=False)
    )
    if directional_auc >= 0.60 and minimum_recall >= best_endpoint_minimum_recall + 0.05:
        label = "SUPPORTS_DIRECTIONAL_SIGNAL"
    elif filter_auc >= 0.60 and directional_auc < 0.56:
        label = "FILTER_ONLY_SIGNAL"
    elif filter_auc < 0.56 and directional_auc < 0.56:
        label = "NO_USEFUL_SIGNAL"
    else:
        label = "INCONCLUSIVE"
    return {
        "verdict": label,
        "reason": (
            f"Validation BUY-vs-SELL AUC={directional_auc:.4f}, directional-vs-NO_TRADE AUC={filter_auc:.4f}, "
            f"minimum BUY/SELL recall={minimum_recall:.4f}, best endpoint minimum recall={best_endpoint_minimum_recall:.4f}."
        ),
    }


def render_report(report: dict[str, Any], comparison: pd.DataFrame) -> str:
    validation = report["evaluation"]["validation"]["multiclass"]
    lines = [
        f"# {EXPERIMENT_ID}", "", "- Target-only controlled experiment", "- TEST rows/labels exposed: 0 / false", "",
        "## Target", "- Entry: current Close", "- Horizon: five future M15 bars", "- Barriers: entry +/- 1.0 x ATR[t]", "- SAME_BAR_TIE rows excluded", "",
        "## Distribution Before Tie Exclusion", "| Split | Class | Count | Percentage |", "|---|---|---:|---:|",
    ]
    for split, rows in report["target_distribution_before_exclusion"].items():
        for row in rows:
            lines.append(f"| {split.upper()} | {row['class']} | {row['count']:,} | {row['percentage']:.2f}% |")
    lines.extend(["", "## Distribution After Tie Exclusion", "| Split | Class | Count | Percentage |", "|---|---|---:|---:|"])
    for split, rows in report["target_distribution_after_exclusion"].items():
        for row in rows:
            lines.append(f"| {split.upper()} | {row['class']} | {row['count']:,} | {row['percentage']:.2f}% |")
    lines.extend(["", "## Validation Comparison", "| Model | Accuracy | Balanced acc. | Macro F1 | SELL P/R/F1 | BUY P/R/F1 | NO_TRADE P/R/F1 | Pred S/B/N | Direction AUC | Filter AUC | CV accuracy | CV macro F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparison.itertuples(index=False):
        direction_auc = "n/a" if pd.isna(row.direction_roc_auc) else f"{row.direction_roc_auc:.4f}"
        filter_auc = "n/a" if pd.isna(row.filter_roc_auc) else f"{row.filter_roc_auc:.4f}"
        cv_macro_f1 = "not recorded" if pd.isna(row.cv_macro_f1_mean) else f"{row.cv_macro_f1_mean:.4f}"
        lines.append(
            f"| {row.model} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.sell_precision:.4f}/{row.sell_recall:.4f}/{row.sell_f1:.4f} | "
            f"{row.buy_precision:.4f}/{row.buy_recall:.4f}/{row.buy_f1:.4f} | "
            f"{row.no_trade_precision:.4f}/{row.no_trade_recall:.4f}/{row.no_trade_f1:.4f} | "
            f"{row.sell_prediction_pct:.2f}%/{row.buy_prediction_pct:.2f}%/{row.no_trade_prediction_pct:.2f}% | {direction_auc} | {filter_auc} | "
            f"{row.cv_accuracy_mean:.4f} | {cv_macro_f1} |"
        )
    lines.extend(["", "## Confusion Matrix", "Rows are actual SELL, BUY, NO_TRADE; columns are predicted SELL, BUY, NO_TRADE.", "", "```text", *[str(row) for row in validation["confusion_matrix"]], "```", "", "## Five-fold TRAIN-only CV", "| Fold | Accuracy | Macro F1 | Direction AUC | Filter AUC |", "|---:|---:|---:|---:|---:|"])
    for fold in report["cv"]["folds"]:
        lines.append(f"| {fold['fold']} | {fold['accuracy']:.4f} | {fold['macro_f1']:.4f} | {fold['direction_roc_auc']:.4f} | {fold['filter_roc_auc']:.4f} |")
    lines.extend(["", "## Verdict", f"**{report['hypothesis']['verdict']}**: {report['hypothesis']['reason']}"])
    return "\n".join(lines) + "\n"


def run_experiment(
    output_dir: Path = OUTPUT_DIR, *, recover_saved_model: bool = False
) -> dict[str, Any]:
    expected_model_path = output_dir / f"{EXPERIMENT_ID}.joblib"
    if recover_saved_model:
        existing_files = sorted(path.name for path in output_dir.iterdir()) if output_dir.is_dir() else []
        if existing_files != [expected_model_path.name]:
            raise RuntimeError(
                "Recovery requires an output directory containing only the already-fitted experiment model"
            )
    else:
        ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    features = list(baseline_report["features"]["names_in_order"])
    if features != list(metadata["feature_names"]) or len(features) != 42:
        raise RuntimeError("Exact reference 42-feature order is required")
    reference_reports = {
        name: json.loads(spec["report"].read_text(encoding="utf-8"))
        for name, spec in REFERENCE_SPECS.items()
    }
    directional_report = json.loads(DIRECTIONAL_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    protected_before = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        **{f"{name}_model_sha256": sha256(spec["model"]) for name, spec in REFERENCE_SPECS.items()},
        **{f"{name}_report_sha256": sha256(spec["report"]) for name, spec in REFERENCE_SPECS.items()},
        "directional_diagnostic_sha256": sha256(DIRECTIONAL_DIAGNOSTIC_PATH),
    }
    requested_columns = list(dict.fromkeys([*features, *OHLC_RECONSTRUCTION_COLUMNS, "target"]))
    train_raw, validation_raw, access = load_train_validation_only(DATASET_PATH, metadata, requested_columns)
    exposed = pd.concat([train_raw, validation_raw])
    ohlc = reconstruct_ohlc(exposed)
    target_rows = build_first_passage_target(ohlc, exposed["atr"])
    independent = independently_reconstruct_target(ohlc, exposed["atr"])
    if not np.array_equal(target_rows["first_passage_target"].to_numpy(), independent):
        raise RuntimeError("Independent target reconstruction failed")
    train, validation = prepare_splits(train_raw, validation_raw, target_rows)
    if set(train["target"].unique()) != {0, 1, 2} or set(validation["target"].unique()) != {0, 1, 2}:
        raise RuntimeError("First-passage TRAIN and VALIDATION must each contain all three classes")
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("Sealed loader exposed TEST data")
    if int((target_rows.loc[validation_raw.index, "first_passage_target"] == INCOMPLETE_LABEL).sum()) != HORIZON:
        raise RuntimeError("Exactly the final five VALIDATION entries must remain incomplete")
    parameters = model_parameters(baseline_report)
    model = XGBoostModel(**parameters, models_dir=str(output_dir))
    if recover_saved_model:
        model.load(EXPERIMENT_ID)
        training_seconds = None
    else:
        training_started = time.perf_counter()
        model.fit_training_data(train, features)
        training_seconds = time.perf_counter() - training_started
    if int(model.scaler.n_samples_seen_) != len(train):
        raise RuntimeError("Main scaler was not fit on exactly the retained TRAIN rows")
    if model.feature_columns != features:
        raise RuntimeError("Main model does not preserve the exact 42-feature order")
    evaluation = {"train": evaluate_model(model, train), "validation": evaluate_model(model, validation)}
    cv_started = time.perf_counter()
    cv_results, cv_details = chronological_cv(train, features, parameters)
    cv_seconds = time.perf_counter() - cv_started
    comparison = comparison_table(
        reference_reports, directional_report, evaluation["validation"], cv_results
    )
    hypothesis = verdict(comparison)
    model_path = expected_model_path if recover_saved_model else Path(model.save(EXPERIMENT_ID))
    reloaded = XGBoostModel(models_dir=str(output_dir))
    reloaded.load(EXPERIMENT_ID)
    values = validation[features].to_numpy()
    predictions = model.model.predict(model.scaler.transform(values))
    reloaded_predictions = reloaded.model.predict(reloaded.scaler.transform(values))
    if not np.array_equal(predictions, reloaded_predictions) or reloaded.feature_columns != features:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("Saved model reload did not preserve predictions and feature order")
    protected_after = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        **{f"{name}_model_sha256": sha256(spec["model"]) for name, spec in REFERENCE_SPECS.items()},
        **{f"{name}_report_sha256": sha256(spec["report"]) for name, spec in REFERENCE_SPECS.items()},
        "directional_diagnostic_sha256": sha256(DIRECTIONAL_DIAGNOSTIC_PATH),
    }
    if protected_before != protected_after:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("Protected reference artifact changed")
    train_targets_before = target_rows.loc[train_raw.index, "first_passage_target"]
    validation_targets_before = target_rows.loc[validation_raw.index, "first_passage_target"]
    tie_counts = {
        "train": int((train_targets_before == TIE_LABEL).sum()),
        "validation": int((validation_targets_before == TIE_LABEL).sum()),
    }
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "training_seconds": training_seconds,
        "cv_seconds": cv_seconds,
        "report_recovery": {
            "used": recover_saved_model,
            "final_model_refit": False if recover_saved_model else None,
            "reason": "resume after directory creation collision during artifact emission" if recover_saved_model else None,
        },
        "target_definition": {
            "entry": "current Close", "horizon_bars": HORIZON, "horizon_minutes": 75,
            "barrier_atr_multiplier": BARRIER_MULTIPLIER, "tie_policy": "exclude",
            "unresolved_policy": "NO_TRADE", "spread_used": False, "point_value_used": False,
        },
        "features": features,
        "model_parameters": {**parameters, "objective": "multi:softprob", "eval_metric": "mlogloss", "early_stopping": False},
        "boundaries": {
            "original_train_rows": len(train_raw), "original_validation_rows": len(validation_raw),
            "train_first": pd.Timestamp(train_raw.index[0]).isoformat(), "train_last": pd.Timestamp(train_raw.index[-1]).isoformat(),
            "validation_first": pd.Timestamp(validation_raw.index[0]).isoformat(), "validation_last": pd.Timestamp(validation_raw.index[-1]).isoformat(),
            "post_exclusion_train_rows": len(train), "post_exclusion_validation_rows": len(validation),
        },
        "data_access": access,
        "target_reconstruction": {
            "independent_match": True, "reconstructed_from": OHLC_RECONSTRUCTION_COLUMNS,
            "complete_rows": int((target_rows["first_passage_target"] != INCOMPLETE_LABEL).sum()),
            "incomplete_validation_tail_rows": int((validation_targets_before == INCOMPLETE_LABEL).sum()),
        },
        "same_bar_ties": {
            **tie_counts,
            "total": sum(tie_counts.values()),
            "percentage_of_complete_rows": float(100 * sum(tie_counts.values()) / (len(exposed) - HORIZON)),
        },
        "target_distribution_before_exclusion": {
            "train": target_distribution(train_targets_before),
            "validation": target_distribution(validation_targets_before),
        },
        "target_distribution_after_exclusion": {
            "train": class_distribution(train), "validation": class_distribution(validation),
        },
        "preprocessing": {"type": "StandardScaler", "fit_scope": "post-tie-exclusion TRAIN only", "scaler_n_samples_seen": int(model.scaler.n_samples_seen_)},
        "evaluation": evaluation,
        "cv": {"scope": "post-tie-exclusion TRAIN only", "folds": cv_details},
        "comparison": comparison.astype(object).where(pd.notna(comparison), None).to_dict(orient="records"),
        "hypothesis": hypothesis,
        "integrity": {
            "before": protected_before, "after": protected_after,
            "protected_artifacts_unchanged": protected_before == protected_after,
            "test_rows_exposed": 0, "test_labels_read": False, "test_evaluated": False,
            "reload_predictions_identical": bool(np.array_equal(predictions, reloaded_predictions)),
            "reload_feature_order_preserved": reloaded.feature_columns == features,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target_audit = target_rows.copy()
    target_audit["split"] = ["train"] * len(train_raw) + ["validation"] * len(validation_raw)
    target_audit.to_csv(output_dir / "target_audit.csv")
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    pd.DataFrame(evaluation["validation"]["multiclass"]["confusion_matrix"], index=["actual_sell", "actual_buy", "actual_no_trade"], columns=["predicted_sell", "predicted_buy", "predicted_no_trade"]).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "experiment_report.md").write_text(render_report(report, comparison), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["hypothesis"], indent=2))