"""Run the controlled normalized/relative-only 18-feature ablation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_baseline import load_train_validation_only, sha256
from run_ablation_25f import (
    BASELINE_DIR,
    BASELINE_ID,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DATASET_PATH,
    DIAGNOSTIC_REPORT_PATH,
    METADATA_PATH,
    assert_reference_hashes,
    chronological_cv,
    ensure_output_available,
    evaluate_fitted_model,
    importance_concentration,
    model_parameters,
)
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_ablation_18f_normalized"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
ABLATION_25F_DIR = BASELINE_DIR / f"{BASELINE_ID}_ablation_25f"
ABLATION_25F_REPORT_PATH = ABLATION_25F_DIR / "experiment_report.json"
ABLATION_25F_MODEL_PATH = ABLATION_25F_DIR / f"{BASELINE_ID}_ablation_25f.joblib"


def resolve_normalized_features(
    diagnostic_report: dict[str, Any], metadata: dict[str, Any]
) -> list[str]:
    features = list(
        diagnostic_report.get("feature_families", {}).get("normalized_relative", [])
    )
    if len(features) != 18 or len(set(features)) != 18:
        raise ValueError("Diagnostic normalized family must contain exactly 18 unique features")
    metadata_features = list(metadata.get("feature_names", []))
    missing = [feature for feature in features if feature not in metadata_features]
    if missing:
        raise ValueError(f"Normalized features absent from dataset metadata: {missing}")
    expected_order = [feature for feature in metadata_features if feature in set(features)]
    if features != expected_order:
        raise ValueError("Normalized feature order differs from dataset metadata")
    return features


def _summary_metrics(report: dict[str, Any], cv_mean: dict[str, float]) -> dict[str, float]:
    train = report["evaluation"]["train"]
    validation = report["evaluation"]["validation"]
    return {
        "validation_accuracy": validation["accuracy"],
        "validation_balanced_accuracy": validation["balanced_accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "buy_recall": validation["per_class"]["BUY"]["recall"],
        "sell_recall": validation["per_class"]["SELL"]["recall"],
        "no_trade_recall": validation["per_class"]["NO_TRADE"]["recall"],
        "buy_prediction_pct": validation["prediction_distribution"]["BUY"]["percentage"],
        "sell_prediction_pct": validation["prediction_distribution"]["SELL"]["percentage"],
        "no_trade_prediction_pct": validation["prediction_distribution"]["NO_TRADE"]["percentage"],
        "accuracy_gap": train["accuracy"] - validation["accuracy"],
        "balanced_accuracy_gap": train["balanced_accuracy"] - validation["balanced_accuracy"],
        "macro_f1_gap": train["macro_f1"] - validation["macro_f1"],
        "cv_mean_accuracy": cv_mean["accuracy"],
        "cv_mean_macro_f1": cv_mean["macro_f1"],
    }


def three_model_comparison(
    baseline_report: dict[str, Any],
    diagnostic_report: dict[str, Any],
    ablation_25f_report: dict[str, Any],
    evaluation_18f: dict[str, Any],
    cv_results_18f: pd.DataFrame,
) -> pd.DataFrame:
    baseline_cv = {
        "accuracy": float(np.mean([fold["metrics"]["accuracy"] for fold in diagnostic_report["importance_fold_details"]])),
        "macro_f1": float(np.mean([fold["metrics"]["macro_f1"] for fold in diagnostic_report["importance_fold_details"]])),
    }
    ablation_25f_cv_rows = ablation_25f_report["cv"]["folds"]
    ablation_25f_cv = {
        "accuracy": float(np.mean([fold["accuracy"] for fold in ablation_25f_cv_rows])),
        "macro_f1": float(np.mean([fold["macro_f1"] for fold in ablation_25f_cv_rows])),
    }
    mean_18f = cv_results_18f[cv_results_18f["row_type"] == "mean"].iloc[0]
    ablation_18f_cv = {
        "accuracy": float(mean_18f["accuracy"]),
        "macro_f1": float(mean_18f["macro_f1"]),
    }
    reference = _summary_metrics(baseline_report, baseline_cv)
    ablation_25f = _summary_metrics(ablation_25f_report, ablation_25f_cv)
    ablation_18f = _summary_metrics(
        {"evaluation": evaluation_18f}, ablation_18f_cv
    )
    labels = {
        "validation_accuracy": "Validation Accuracy",
        "validation_balanced_accuracy": "Validation Balanced Accuracy",
        "validation_macro_f1": "Validation Macro F1",
        "buy_recall": "BUY Recall",
        "sell_recall": "SELL Recall",
        "no_trade_recall": "NO_TRADE Recall",
        "buy_prediction_pct": "BUY Prediction %",
        "sell_prediction_pct": "SELL Prediction %",
        "no_trade_prediction_pct": "NO_TRADE Prediction %",
        "accuracy_gap": "TRAIN-VALIDATION Accuracy Gap",
        "balanced_accuracy_gap": "TRAIN-VALIDATION Balanced Accuracy Gap",
        "macro_f1_gap": "TRAIN-VALIDATION Macro F1 Gap",
        "cv_mean_accuracy": "CV Mean Accuracy",
        "cv_mean_macro_f1": "CV Mean Macro F1",
    }
    return pd.DataFrame(
        [
            {
                "metric": label,
                "reference_42f": reference[key],
                "ablation_25f": ablation_25f[key],
                "normalized_18f": ablation_18f[key],
                "difference_18f_vs_42f": ablation_18f[key] - reference[key],
                "difference_18f_vs_25f": ablation_18f[key] - ablation_25f[key],
            }
            for key, label in labels.items()
        ]
    )


def signal_verdict(comparison: pd.DataFrame) -> tuple[str, str]:
    values = comparison.set_index("metric")
    balanced = values.loc["Validation Balanced Accuracy", "normalized_18f"]
    macro_f1 = values.loc["Validation Macro F1", "normalized_18f"]
    cv_macro = values.loc["CV Mean Macro F1", "normalized_18f"]
    buy_recall = values.loc["BUY Recall", "normalized_18f"]
    sell_recall = values.loc["SELL Recall", "normalized_18f"]
    majority_balanced = 1 / 3
    majority_macro_f1 = 0.24755461149135918
    if (
        balanced > majority_balanced + 0.02
        and macro_f1 > majority_macro_f1 + 0.05
        and cv_macro > majority_macro_f1 + 0.05
        and buy_recall > 0.10
        and sell_recall > 0.10
    ):
        return (
            "SUPPORTS",
            "Normalized/relative features retain reproducible class information beyond the majority baseline in both holdout and chronological CV.",
        )
    if (
        balanced <= majority_balanced + 0.01
        and macro_f1 <= majority_macro_f1 + 0.03
        and cv_macro <= majority_macro_f1 + 0.03
    ):
        return (
            "REFUTES",
            "Normalized/relative features do not show material class information beyond the majority baseline.",
        )
    return (
        "INCONCLUSIVE",
        "Normalized/relative features show stable above-majority CV signal, but weak BUY/SELL recall and lower class-balanced holdout metrics prevent a clear conclusion.",
    )


def render_report(report: dict[str, Any], comparison: pd.DataFrame) -> str:
    evaluation = report["evaluation"]
    validation = evaluation["validation"]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- Status: {report['status']}",
        f"- Runtime: {report['runtime_seconds']:.2f} seconds",
        "- TEST labels read/evaluated: false / false",
        "",
        "## Exact Ordered Features",
        *[f"{index}. `{feature}`" for index, feature in enumerate(report["features"], 1)],
        "",
        "## TRAIN and VALIDATION",
        "| Split | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
        f"| TRAIN | {evaluation['train']['accuracy']:.4f} | {evaluation['train']['balanced_accuracy']:.4f} | {evaluation['train']['macro_f1']:.4f} |",
        f"| VALIDATION | {validation['accuracy']:.4f} | {validation['balanced_accuracy']:.4f} | {validation['macro_f1']:.4f} |",
        "",
        "## Three-model Comparison",
        "| Metric | Reference 42f | Ablation 25f | Normalized 18f | 18f-42f | 18f-25f |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.reference_42f:.4f} | {row.ablation_25f:.4f} | "
            f"{row.normalized_18f:.4f} | {row.difference_18f_vs_42f:+.4f} | "
            f"{row.difference_18f_vs_25f:+.4f} |"
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
            "## Native Feature Importance",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
            *[
                f"| {item['rank']} | {item['feature']} | {item['importance']:.6f} |"
                for item in report["feature_importance"]
            ],
            "",
            "## Concentration",
            f"- 42f top-1/top-2/entropy: {report['importance_concentration']['reference_42f']['top1_share']:.4f} / {report['importance_concentration']['reference_42f']['top2_share']:.4f} / {report['importance_concentration']['reference_42f']['normalized_entropy']:.4f}",
            f"- 25f top-1/top-2/entropy: {report['importance_concentration']['ablation_25f']['top1_share']:.4f} / {report['importance_concentration']['ablation_25f']['top2_share']:.4f} / {report['importance_concentration']['ablation_25f']['normalized_entropy']:.4f}",
            f"- 18f top-1/top-2/entropy: {report['importance_concentration']['normalized_18f']['top1_share']:.4f} / {report['importance_concentration']['normalized_18f']['top2_share']:.4f} / {report['importance_concentration']['normalized_18f']['normalized_entropy']:.4f}",
            "- Native importance is descriptive and does not establish causality.",
            "",
            "## Verdict",
            f"**{report['hypothesis']['verdict']}**: {report['hypothesis']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    diagnostic_report = json.loads(DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8"))
    ablation_25f_report = json.loads(ABLATION_25F_REPORT_PATH.read_text(encoding="utf-8"))
    reference_hashes = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "reference_model_sha256": sha256(BASELINE_MODEL_PATH),
        "ablation_25f_model_sha256": sha256(ABLATION_25F_MODEL_PATH),
    }
    features = resolve_normalized_features(diagnostic_report, metadata)
    parameters = model_parameters(baseline_report)
    train, validation, access = load_train_validation_only(
        DATASET_PATH, metadata, [*features, "target"]
    )
    if access["rows_exposed"] != 79908:
        raise RuntimeError("Experiment must expose exactly 79,908 rows")
    if access["last_exposed_timestamp"] != metadata["splits"]["validation"]["last_timestamp"]:
        raise RuntimeError("Exposed data exceeds or misses VALIDATION boundary")

    model = XGBoostModel(**parameters, models_dir=str(output_dir))
    training_started = time.perf_counter()
    model.fit_training_data(train, features)
    training_seconds = time.perf_counter() - training_started
    if int(model.scaler.n_samples_seen_) != 71917:
        raise RuntimeError("Scaler must fit exactly 71,917 TRAIN rows")
    evaluation = evaluate_fitted_model(model, train, validation, features)
    cv_started = time.perf_counter()
    cv_results, cv_details = chronological_cv(train, features, parameters)
    cv_seconds = time.perf_counter() - cv_started
    importance = model.feature_importance().copy()
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    comparison = three_model_comparison(
        baseline_report,
        diagnostic_report,
        ablation_25f_report,
        evaluation,
        cv_results,
    )
    verdict, reason = signal_verdict(comparison)

    model_path = Path(model.save(EXPERIMENT_ID))
    reloaded = XGBoostModel(models_dir=str(output_dir))
    reloaded.load(EXPERIMENT_ID)
    X_validation = validation[features].to_numpy()
    predictions = model.model.predict(model.scaler.transform(X_validation))
    reloaded_predictions = reloaded.model.predict(reloaded.scaler.transform(X_validation))
    assert_reference_hashes(
        {key: reference_hashes[key] for key in ("dataset_sha256", "metadata_sha256", "reference_model_sha256")}
    )
    if sha256(ABLATION_25F_MODEL_PATH) != reference_hashes["ablation_25f_model_sha256"]:
        raise RuntimeError("25-feature reference artifact changed")

    concentration = {
        "reference_42f": importance_concentration(
            [float(item["importance"]) for item in baseline_report["feature_importance"]]
        ),
        "ablation_25f": importance_concentration(
            [float(item["importance"]) for item in ablation_25f_report["feature_importance"]]
        ),
        "normalized_18f": importance_concentration(
            importance["importance"].astype(float).tolist()
        ),
    }
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
            "validation_end_timestamp": metadata["splits"]["validation"]["last_timestamp"],
        },
        "data_access": access,
        "preprocessing": {
            "type": "StandardScaler",
            "fit_scope": "TRAIN only",
            "scaler_n_samples_seen": int(model.scaler.n_samples_seen_),
        },
        "evaluation": evaluation,
        "gaps": {
            "accuracy": evaluation["train"]["accuracy"] - evaluation["validation"]["accuracy"],
            "balanced_accuracy": evaluation["train"]["balanced_accuracy"] - evaluation["validation"]["balanced_accuracy"],
            "macro_f1": evaluation["train"]["macro_f1"] - evaluation["validation"]["macro_f1"],
        },
        "cv": {"scope": "TRAIN only", "folds": cv_details},
        "feature_importance": importance.to_dict(orient="records"),
        "importance_concentration": concentration,
        "comparison": comparison.to_dict(orient="records"),
        "hypothesis": {"verdict": verdict, "reason": reason},
        "integrity": {
            **reference_hashes,
            "dataset_sha256_after": sha256(DATASET_PATH),
            "metadata_sha256_after": sha256(METADATA_PATH),
            "reference_model_sha256_after": sha256(BASELINE_MODEL_PATH),
            "ablation_25f_model_sha256_after": sha256(ABLATION_25F_MODEL_PATH),
            "dataset_unchanged": sha256(DATASET_PATH) == reference_hashes["dataset_sha256"],
            "metadata_unchanged": sha256(METADATA_PATH) == reference_hashes["metadata_sha256"],
            "reference_model_unchanged": sha256(BASELINE_MODEL_PATH) == reference_hashes["reference_model_sha256"],
            "ablation_25f_model_unchanged": sha256(ABLATION_25F_MODEL_PATH) == reference_hashes["ablation_25f_model_sha256"],
            "test_labels_read": False,
            "test_evaluated": False,
            "reload_predictions_identical": bool(np.array_equal(predictions, reloaded_predictions)),
            "reload_feature_order_preserved": reloaded.feature_columns == features,
        },
    }
    if not all(
        value is True
        for key, value in report["integrity"].items()
        if key.endswith("_unchanged") or key.startswith("reload_")
    ) or report["integrity"]["test_labels_read"] or report["integrity"]["test_evaluated"]:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Experiment integrity failure: {report['integrity']}")

    confusion_rows = []
    for split in ("train", "validation"):
        matrix = evaluation[split]["confusion_matrix"]
        for index, actual in enumerate(("SELL", "BUY", "NO_TRADE")):
            confusion_rows.append(
                {
                    "split": split,
                    "actual": actual,
                    "predicted_sell": matrix[index][0],
                    "predicted_buy": matrix[index][1],
                    "predicted_no_trade": matrix[index][2],
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