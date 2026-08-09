"""Run the single controlled 25-feature V0.3 XGBoost ablation."""

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
from src.model import XGBoostModel
from train_baseline import BASELINE_ID, evaluation_metrics


EXPERIMENT_ID = f"{BASELINE_ID}_ablation_25f"
BASELINE_DIR = Path("models/baselines")
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
DATASET_PATH = Path(
    "data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.parquet"
)
METADATA_PATH = DATASET_PATH.with_suffix(".metadata.json")
BASELINE_MODEL_PATH = BASELINE_DIR / f"{BASELINE_ID}.joblib"
BASELINE_REPORT_PATH = BASELINE_DIR / f"{BASELINE_ID}_evaluation.json"
DIAGNOSTIC_REPORT_PATH = (
    BASELINE_DIR / f"{BASELINE_ID}_diagnostics" / "diagnostic_report.json"
)

EXPECTED_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Experiment output already exists: {output_dir}")


def resolve_feature_set(
    diagnostic_report: dict[str, Any], metadata: dict[str, Any]
) -> list[str]:
    """Resolve the ordered 18+7 feature union from recorded diagnostics."""
    families = diagnostic_report.get("feature_families", {})
    normalized = list(families.get("normalized_relative", []))
    session_time = list(families.get("session_time", []))
    if len(normalized) != 18 or len(session_time) != 7:
        raise ValueError(
            "Diagnostic feature families must contain exactly 18 normalized and 7 session/time features"
        )
    features = [*normalized, *session_time]
    if len(features) != 25 or len(set(features)) != 25:
        raise ValueError("Ablation feature set must contain exactly 25 unique features")
    metadata_features = list(metadata.get("feature_names", []))
    missing = [feature for feature in features if feature not in metadata_features]
    if missing:
        raise ValueError(f"Diagnostic features are absent from dataset metadata: {missing}")
    expected_order = [feature for feature in metadata_features if feature in set(features)]
    if expected_order != features:
        raise ValueError(
            "Diagnostic feature order does not match dataset metadata order; refusing to reorder"
        )
    return features


def assert_reference_hashes(
    expected: dict[str, str],
    *,
    dataset_path: Path = DATASET_PATH,
    metadata_path: Path = METADATA_PATH,
    baseline_model_path: Path = BASELINE_MODEL_PATH,
) -> dict[str, str]:
    actual = {
        "dataset_sha256": sha256(dataset_path),
        "metadata_sha256": sha256(metadata_path),
        "reference_model_sha256": sha256(baseline_model_path),
    }
    if actual != expected:
        raise RuntimeError(f"Reference artifact hash changed: expected={expected}, actual={actual}")
    return actual


def model_parameters(reference_report: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        key: reference_report["model_parameters"][key]
        for key in EXPECTED_PARAMETERS
    }
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(
            f"Reference model parameters differ from controlled experiment: {parameters}"
        )
    if reference_report["model_parameters"].get("objective") != "multi:softprob":
        raise ValueError("Reference objective must be multi:softprob")
    if reference_report["model_parameters"].get("eval_metric") != "mlogloss":
        raise ValueError("Reference effective evaluation metric must be mlogloss")
    return parameters


def evaluate_fitted_model(
    model: XGBoostModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    result = {}
    for split, frame in (("train", train), ("validation", validation)):
        values = frame[features].to_numpy()
        predictions = model.model.predict(model.scaler.transform(values))
        result[split] = evaluation_metrics(frame["target"].to_numpy(), predictions)
    return result


def chronological_cv(
    train: pd.DataFrame,
    features: list[str],
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    details = []
    for fold, (train_indices, validation_indices) in enumerate(
        TimeSeriesSplit(n_splits=5).split(train), 1
    ):
        fold_train = train.iloc[train_indices]
        fold_validation = train.iloc[validation_indices]
        if fold_train.index[-1] >= fold_validation.index[0]:
            raise RuntimeError("Chronological ordering violated in CV fold")
        model = XGBoostModel(**parameters, models_dir=str(BASELINE_DIR))
        model.fit_training_data(fold_train, features)
        if int(model.scaler.n_samples_seen_) != len(fold_train):
            raise RuntimeError("Fold scaler was not fit on exactly the fold TRAIN rows")
        metrics = evaluate_fitted_model(
            model, fold_train, fold_validation, features
        )["validation"]
        row = {
            "row_type": "fold",
            "fold": fold,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "buy_recall": metrics["per_class"]["BUY"]["recall"],
            "scaler_fit_rows": int(model.scaler.n_samples_seen_),
        }
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
    fold_frame = pd.DataFrame(rows)
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "buy_recall"]
    summary_rows = []
    for statistic in ("mean", "std", "min", "max"):
        values = getattr(fold_frame[metric_columns], statistic)(ddof=0) if statistic == "std" else getattr(fold_frame[metric_columns], statistic)()
        summary_rows.append(
            {"row_type": statistic, "fold": np.nan, **values.to_dict(), "scaler_fit_rows": np.nan}
        )
    return pd.concat([fold_frame, pd.DataFrame(summary_rows)], ignore_index=True), details


def _metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "validation_accuracy": metrics["accuracy"],
        "validation_balanced_accuracy": metrics["balanced_accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "buy_recall": metrics["per_class"]["BUY"]["recall"],
        "sell_recall": metrics["per_class"]["SELL"]["recall"],
        "buy_prediction_pct": metrics["prediction_distribution"]["BUY"]["percentage"],
        "sell_prediction_pct": metrics["prediction_distribution"]["SELL"]["percentage"],
        "no_trade_prediction_pct": metrics["prediction_distribution"]["NO_TRADE"]["percentage"],
    }


def comparison_table(
    baseline_report: dict[str, Any],
    diagnostic_report: dict[str, Any],
    ablation_metrics: dict[str, Any],
    ablation_cv: pd.DataFrame,
) -> pd.DataFrame:
    baseline_validation = baseline_report["evaluation"]["validation"]
    baseline_values = _metric_row(baseline_validation)
    ablation_values = _metric_row(ablation_metrics["validation"])
    baseline_folds = diagnostic_report["importance_fold_details"]
    baseline_values["cv_mean_accuracy"] = float(
        np.mean([fold["metrics"]["accuracy"] for fold in baseline_folds])
    )
    baseline_values["cv_mean_macro_f1"] = float(
        np.mean([fold["metrics"]["macro_f1"] for fold in baseline_folds])
    )
    ablation_mean = ablation_cv[ablation_cv["row_type"] == "mean"].iloc[0]
    ablation_values["cv_mean_accuracy"] = float(ablation_mean["accuracy"])
    ablation_values["cv_mean_macro_f1"] = float(ablation_mean["macro_f1"])
    baseline_gap = baseline_report["train_validation_gap"]
    ablation_values.update(
        {
            "accuracy_gap": ablation_metrics["train"]["accuracy"] - ablation_metrics["validation"]["accuracy"],
            "balanced_accuracy_gap": ablation_metrics["train"]["balanced_accuracy"] - ablation_metrics["validation"]["balanced_accuracy"],
            "macro_f1_gap": ablation_metrics["train"]["macro_f1"] - ablation_metrics["validation"]["macro_f1"],
        }
    )
    baseline_values.update(
        {
            "accuracy_gap": baseline_gap["accuracy"],
            "balanced_accuracy_gap": baseline_gap["balanced_accuracy"],
            "macro_f1_gap": baseline_gap["macro_f1"],
        }
    )
    labels = {
        "validation_accuracy": "Validation Accuracy",
        "validation_balanced_accuracy": "Validation Balanced Accuracy",
        "validation_macro_f1": "Validation Macro F1",
        "buy_recall": "BUY Recall",
        "sell_recall": "SELL Recall",
        "buy_prediction_pct": "BUY Prediction %",
        "sell_prediction_pct": "SELL Prediction %",
        "no_trade_prediction_pct": "NO_TRADE Prediction %",
        "cv_mean_accuracy": "CV Mean Accuracy",
        "cv_mean_macro_f1": "CV Mean Macro F1",
        "accuracy_gap": "TRAIN-VALIDATION Accuracy Gap",
        "balanced_accuracy_gap": "TRAIN-VALIDATION Balanced Accuracy Gap",
        "macro_f1_gap": "TRAIN-VALIDATION Macro F1 Gap",
    }
    return pd.DataFrame(
        [
            {
                "metric": labels[key],
                "baseline": baseline_values[key],
                "ablation_25f": ablation_values[key],
                "difference": ablation_values[key] - baseline_values[key],
            }
            for key in labels
        ]
    )


def hypothesis_verdict(comparison: pd.DataFrame) -> tuple[str, str]:
    values = comparison.set_index("metric")
    macro_gain = values.loc["Validation Macro F1", "difference"]
    balanced_gain = values.loc["Validation Balanced Accuracy", "difference"]
    accuracy_gain = values.loc["Validation Accuracy", "difference"]
    gap_reduction = -values.loc["TRAIN-VALIDATION Macro F1 Gap", "difference"]
    buy_gain = values.loc["BUY Recall", "difference"]
    cv_gain = values.loc["CV Mean Macro F1", "difference"]
    if (
        macro_gain > 0
        and balanced_gain > 0
        and gap_reduction > 0
        and buy_gain > 0
        and cv_gain >= -0.01
        and accuracy_gain > -0.03
    ):
        return "SUPPORTED", "Validation class balance, BUY recall, and generalization gap improved without material CV or accuracy damage."
    if macro_gain < -0.02 and balanced_gain < -0.02 and gap_reduction <= 0:
        return "WEAKENED", "Removing scale-dependent features degraded class-balanced validation metrics without reducing overfit."
    return "INCONCLUSIVE", "The measured changes are mixed or insufficiently consistent across validation, gaps, directional behavior, and CV."


def importance_concentration(importances: list[float]) -> dict[str, float]:
    values = np.asarray(importances, dtype=float)
    probabilities = values / values.sum()
    positive = probabilities[probabilities > 0]
    return {
        "top1_share": float(np.max(probabilities)),
        "top2_share": float(np.sort(probabilities)[-2:].sum()),
        "hhi": float(np.square(probabilities).sum()),
        "normalized_entropy": float(
            -np.sum(positive * np.log(positive)) / np.log(len(probabilities))
        ),
    }


def render_report(report: dict[str, Any], comparison: pd.DataFrame) -> str:
    evaluation = report["evaluation"]
    validation = evaluation["validation"]
    importance = report["feature_importance"][:15]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- Status: {report['status']}",
        f"- Runtime: {report['runtime_seconds']:.2f} seconds",
        f"- Features: {len(report['features'])}",
        "- TEST labels read/evaluated: false / false",
        "",
        "## TRAIN and VALIDATION",
        "| Split | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
        f"| TRAIN | {evaluation['train']['accuracy']:.4f} | {evaluation['train']['balanced_accuracy']:.4f} | {evaluation['train']['macro_f1']:.4f} |",
        f"| VALIDATION | {validation['accuracy']:.4f} | {validation['balanced_accuracy']:.4f} | {validation['macro_f1']:.4f} |",
        "",
        "## Exact Ordered Features",
        *[f"{index}. `{feature}`" for index, feature in enumerate(report["features"], 1)],
        "",
        "## Baseline Comparison",
        "| Metric | Baseline | 25-feature ablation | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.baseline:.4f} | {row.ablation_25f:.4f} | {row.difference:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-class Validation",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, metrics in validation["per_class"].items():
        lines.append(
            f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']:,} |"
        )
    lines.extend(
        [
            "",
            "## Chronological TRAIN-only CV",
            "| Fold | Accuracy | Balanced accuracy | Macro F1 | BUY recall |",
            "|---:|---:|---:|---:|---:|",
            *[
                f"| {fold['fold']} | {fold['accuracy']:.4f} | {fold['balanced_accuracy']:.4f} | {fold['macro_f1']:.4f} | {fold['buy_recall']:.4f} |"
                for fold in report["cv"]["folds"]
            ],
            "",
            "## Top Native Importance",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
            *[
                f"| {item['rank']} | {item['feature']} | {item['importance']:.6f} |"
                for item in importance
            ],
            "",
            "## Importance Concentration",
            f"- Baseline top-1/top-2 share: {report['importance_profile']['baseline_concentration']['top1_share']:.4f} / {report['importance_profile']['baseline_concentration']['top2_share']:.4f}",
            f"- Ablation top-1/top-2 share: {report['importance_profile']['ablation_concentration']['top1_share']:.4f} / {report['importance_profile']['ablation_concentration']['top2_share']:.4f}",
            f"- Baseline/ablation normalized entropy: {report['importance_profile']['baseline_concentration']['normalized_entropy']:.4f} / {report['importance_profile']['ablation_concentration']['normalized_entropy']:.4f}",
            "- Native gain importance is descriptive and does not establish causality.",
            "",
            "## Verdict",
            f"**{report['hypothesis']['verdict']}**: {report['hypothesis']['reason']}",
            "",
            f"Next experiment: {report['next_experiment']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    with METADATA_PATH.open(encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)
    with BASELINE_REPORT_PATH.open(encoding="utf-8") as file_handle:
        baseline_report = json.load(file_handle)
    with DIAGNOSTIC_REPORT_PATH.open(encoding="utf-8") as file_handle:
        diagnostic_report = json.load(file_handle)

    reference_hashes = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "reference_model_sha256": sha256(BASELINE_MODEL_PATH),
    }
    features = resolve_feature_set(diagnostic_report, metadata)
    parameters = model_parameters(baseline_report)
    train, validation, access = load_train_validation_only(
        DATASET_PATH, metadata, [*features, "target"]
    )
    if access["rows_exposed"] != 79908:
        raise RuntimeError("Experiment must expose exactly 79,908 rows")
    expected_validation_end = metadata["splits"]["validation"]["last_timestamp"]
    if access["last_exposed_timestamp"] != expected_validation_end:
        raise RuntimeError("Final exposed timestamp does not match VALIDATION boundary")

    model = XGBoostModel(**parameters, models_dir=str(output_dir))
    training_started = time.perf_counter()
    model.fit_training_data(train, features)
    training_seconds = time.perf_counter() - training_started
    scaler_rows = int(model.scaler.n_samples_seen_)
    if scaler_rows != 71917 or scaler_rows != len(train):
        raise RuntimeError("Scaler must fit exactly 71,917 TRAIN rows")
    evaluation = evaluate_fitted_model(model, train, validation, features)

    cv_started = time.perf_counter()
    cv_results, cv_details = chronological_cv(train, features, parameters)
    cv_seconds = time.perf_counter() - cv_started
    importance = model.feature_importance().copy()
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    comparison = comparison_table(
        baseline_report, diagnostic_report, evaluation, cv_results
    )
    verdict, reason = hypothesis_verdict(comparison)

    model_path = Path(model.save(EXPERIMENT_ID))
    reloaded = XGBoostModel(models_dir=str(output_dir))
    reloaded.load(EXPERIMENT_ID)
    validation_values = validation[features].to_numpy()
    original_predictions = model.model.predict(model.scaler.transform(validation_values))
    reloaded_predictions = reloaded.model.predict(
        reloaded.scaler.transform(validation_values)
    )
    assert_reference_hashes(reference_hashes)

    baseline_importance = baseline_report["feature_importance"]
    baseline_top1 = float(baseline_importance[0]["importance"])
    ablation_top1 = float(importance.iloc[0]["importance"])
    report: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "training_seconds": training_seconds,
        "cv_seconds": cv_seconds,
        "features": features,
        "feature_source": str(DIAGNOSTIC_REPORT_PATH),
        "model_parameters": {
            **parameters,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "early_stopping": False,
        },
        "boundaries": {
            "train": [0, 71917],
            "validation": [71917, 79908],
            "test_start": 79908,
            "recorded_validation_end": expected_validation_end,
        },
        "data_access": access,
        "preprocessing": {
            "type": "StandardScaler",
            "fit_scope": "TRAIN only",
            "scaler_n_samples_seen": scaler_rows,
        },
        "evaluation": evaluation,
        "gaps": {
            "accuracy": evaluation["train"]["accuracy"] - evaluation["validation"]["accuracy"],
            "balanced_accuracy": evaluation["train"]["balanced_accuracy"] - evaluation["validation"]["balanced_accuracy"],
            "macro_f1": evaluation["train"]["macro_f1"] - evaluation["validation"]["macro_f1"],
            "buy_recall": evaluation["train"]["per_class"]["BUY"]["recall"] - evaluation["validation"]["per_class"]["BUY"]["recall"],
            "sell_recall": evaluation["train"]["per_class"]["SELL"]["recall"] - evaluation["validation"]["per_class"]["SELL"]["recall"],
        },
        "cv": {"scope": "TRAIN only", "folds": cv_details},
        "feature_importance": importance.to_dict(orient="records"),
        "importance_profile": {
            "baseline_top_feature": baseline_importance[0],
            "ablation_top_feature": importance.iloc[0].to_dict(),
            "baseline_top1_share": baseline_top1,
            "ablation_top1_share": ablation_top1,
            "top1_share_difference": ablation_top1 - baseline_top1,
            "baseline_concentration": importance_concentration(
                [float(item["importance"]) for item in baseline_importance]
            ),
            "ablation_concentration": importance_concentration(
                importance["importance"].astype(float).tolist()
            ),
        },
        "comparison": comparison.to_dict(orient="records"),
        "hypothesis": {"verdict": verdict, "reason": reason},
        "next_experiment": (
            "Run one controlled session/time-only removal from the 25-feature set, retaining the "
            "18 normalized/relative features and identical parameters, to measure whether stable "
            "calendar activity signal helps or masks directional generalization."
        ),
        "integrity": {
            **reference_hashes,
            "dataset_sha256_after": sha256(DATASET_PATH),
            "metadata_sha256_after": sha256(METADATA_PATH),
            "reference_model_sha256_after": sha256(BASELINE_MODEL_PATH),
            "dataset_unchanged": sha256(DATASET_PATH) == reference_hashes["dataset_sha256"],
            "metadata_unchanged": sha256(METADATA_PATH) == reference_hashes["metadata_sha256"],
            "reference_model_unchanged": sha256(BASELINE_MODEL_PATH) == reference_hashes["reference_model_sha256"],
            "reference_baseline_retrained": False,
            "test_labels_read": False,
            "test_evaluated": False,
            "reload_predictions_identical": bool(np.array_equal(original_predictions, reloaded_predictions)),
            "reload_feature_order_preserved": reloaded.feature_columns == features,
        },
    }
    integrity = report["integrity"]
    required_checks = (
        integrity["dataset_unchanged"],
        integrity["metadata_unchanged"],
        integrity["reference_model_unchanged"],
        integrity["reference_baseline_retrained"] is False,
        integrity["test_labels_read"] is False,
        integrity["test_evaluated"] is False,
        integrity["reload_predictions_identical"],
        integrity["reload_feature_order_preserved"],
    )
    if not all(required_checks):
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Experiment integrity checks failed: {integrity}")

    confusion_rows = []
    for split in ("train", "validation"):
        matrix = evaluation[split]["confusion_matrix"]
        for row_index, actual in enumerate(("SELL", "BUY", "NO_TRADE")):
            confusion_rows.append(
                {
                    "split": split,
                    "actual": actual,
                    "predicted_sell": matrix[row_index][0],
                    "predicted_buy": matrix[row_index][1],
                    "predicted_no_trade": matrix[row_index][2],
                }
            )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrix.csv", index=False)
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "experiment_report.md").write_text(
        render_report(report, comparison), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps({"status": result["status"], "hypothesis": result["hypothesis"]}, indent=2))