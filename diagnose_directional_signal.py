"""Diagnose whether saved V0.3 models learn direction or mostly NO_TRADE filtering."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from diagnose_baseline import load_train_validation_only, sha256
from run_ablation_25f import (
    BASELINE_DIR,
    BASELINE_ID,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DATASET_PATH,
    DIAGNOSTIC_REPORT_PATH,
    METADATA_PATH,
    ensure_output_available,
)
from src.model import XGBoostModel
from train_baseline import evaluation_metrics


DIAGNOSTIC_ID = f"{BASELINE_ID}_directional_signal_diagnostic_v1"
OUTPUT_DIR = BASELINE_DIR / DIAGNOSTIC_ID
MODEL_SPECS = {
    "reference_42f": {
        "id": BASELINE_ID,
        "model": BASELINE_MODEL_PATH,
        "report": BASELINE_REPORT_PATH,
    },
    "ablation_25f": {
        "id": f"{BASELINE_ID}_ablation_25f",
        "model": BASELINE_DIR / f"{BASELINE_ID}_ablation_25f" / f"{BASELINE_ID}_ablation_25f.joblib",
        "report": BASELINE_DIR / f"{BASELINE_ID}_ablation_25f" / "experiment_report.json",
    },
    "normalized_18f": {
        "id": f"{BASELINE_ID}_ablation_18f_normalized",
        "model": BASELINE_DIR / f"{BASELINE_ID}_ablation_18f_normalized" / f"{BASELINE_ID}_ablation_18f_normalized.joblib",
        "report": BASELINE_DIR / f"{BASELINE_ID}_ablation_18f_normalized" / "experiment_report.json",
    },
}
CLASS_NAMES = {0: "SELL", 1: "BUY", 2: "NO_TRADE"}
PAIR_DEFINITIONS = (("BUY", "NO_TRADE"), ("SELL", "NO_TRADE"), ("BUY", "SELL"))


def class_distribution(frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = frame["target"].value_counts().reindex([0, 1, 2], fill_value=0)
    return [
        {
            "class": CLASS_NAMES[label],
            "count": int(counts[label]),
            "percentage": float(100 * counts[label] / len(frame)),
        }
        for label in (0, 1, 2)
    ]


def majority_evaluation(frame: pd.DataFrame) -> dict[str, Any]:
    predictions = np.full(len(frame), 2, dtype=int)
    return evaluation_metrics(frame["target"].to_numpy(), predictions)


def _robust_effect(left: np.ndarray, right: np.ndarray) -> float:
    combined = np.concatenate([left, right])
    scale = np.subtract(*np.percentile(combined, [75, 25]))
    if scale == 0:
        return float(abs(np.median(left) - np.median(right)))
    return float(abs(np.median(left) - np.median(right)) / scale)


def feature_separation(frame: pd.DataFrame, features: list[str], split: str) -> pd.DataFrame:
    rows = []
    labels = {name: label for label, name in CLASS_NAMES.items()}
    for feature in features:
        for left_name, right_name in PAIR_DEFINITIONS:
            left = frame.loc[frame["target"] == labels[left_name], feature].to_numpy(dtype=float)
            right = frame.loc[frame["target"] == labels[right_name], feature].to_numpy(dtype=float)
            ks = ks_2samp(left, right)
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "comparison": f"{left_name}_vs_{right_name}",
                    "left_count": len(left),
                    "right_count": len(right),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "absolute_median_iqr_effect": _robust_effect(left, right),
                }
            )
    return pd.DataFrame(rows)


def summarize_separation(results: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (split, comparison), group in results.groupby(["split", "comparison"], sort=False):
        ranked = group.sort_values("ks_statistic", ascending=False)
        rows.append(
            {
                "split": split,
                "comparison": comparison,
                "median_ks": float(group["ks_statistic"].median()),
                "max_ks": float(group["ks_statistic"].max()),
                "features_ks_at_least_0_1": int((group["ks_statistic"] >= 0.1).sum()),
                "median_absolute_median_iqr_effect": float(group["absolute_median_iqr_effect"].median()),
                "top_features": ranked.head(5)[
                    ["feature", "ks_statistic", "absolute_median_iqr_effect"]
                ].to_dict(orient="records"),
            }
        )
    return rows


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def evaluate_saved_model(
    model: XGBoostModel, frame: pd.DataFrame
) -> dict[str, Any]:
    features = model.feature_columns
    values = model.scaler.transform(frame[features].to_numpy())
    predictions = model.model.predict(values).astype(int)
    probabilities = model.model.predict_proba(values)
    class_positions = {
        int(label): position for position, label in enumerate(model.model.classes_)
    }
    actual = frame["target"].to_numpy(dtype=int)
    actual_directional = (actual != 2).astype(int)
    predicted_directional = (predictions != 2).astype(int)
    directional_probability = 1 - probabilities[:, class_positions[2]]
    directional_rows = actual != 2
    buy_probability = probabilities[:, class_positions[1]]
    sell_probability = probabilities[:, class_positions[0]]
    direction_score = buy_probability[directional_rows] / np.clip(
        buy_probability[directional_rows] + sell_probability[directional_rows], 1e-12, None
    )
    direction_actual = (actual[directional_rows] == 1).astype(int)
    direction_prediction = (predictions[directional_rows] == 1).astype(int)
    emitted = predicted_directional == 1
    correct_emitted_direction = (
        (predictions[emitted] == actual[emitted]) & (actual[emitted] != 2)
    )
    return {
        "multiclass": evaluation_metrics(actual, predictions),
        "directional_filter": {
            **_binary_metrics(actual_directional, predicted_directional),
            "roc_auc": float(roc_auc_score(actual_directional, directional_probability)),
        },
        "buy_vs_sell_on_actual_directional": {
            **_binary_metrics(direction_actual, direction_prediction),
            "roc_auc": float(roc_auc_score(direction_actual, direction_score)),
            "actual_directional_rows": int(directional_rows.sum()),
        },
        "emitted_trade_quality": {
            "emitted_rows": int(emitted.sum()),
            "emitted_percentage": float(100 * emitted.mean()),
            "correct_direction_count": int(correct_emitted_direction.sum()),
            "correct_direction_precision": float(correct_emitted_direction.mean()) if emitted.any() else 0.0,
        },
    }


def model_comparison_rows(
    majority: dict[str, Any], model_results: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for name, result in {"majority_no_trade": {"multiclass": majority}, **model_results}.items():
        metrics = result["multiclass"]
        row = {
            "model": name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "sell_precision": metrics["per_class"]["SELL"]["precision"],
            "sell_recall": metrics["per_class"]["SELL"]["recall"],
            "buy_precision": metrics["per_class"]["BUY"]["precision"],
            "buy_recall": metrics["per_class"]["BUY"]["recall"],
            "no_trade_precision": metrics["per_class"]["NO_TRADE"]["precision"],
            "no_trade_recall": metrics["per_class"]["NO_TRADE"]["recall"],
            "sell_prediction_pct": metrics["prediction_distribution"]["SELL"]["percentage"],
            "buy_prediction_pct": metrics["prediction_distribution"]["BUY"]["percentage"],
            "no_trade_prediction_pct": metrics["prediction_distribution"]["NO_TRADE"]["percentage"],
            "filter_roc_auc": None,
            "direction_roc_auc": None,
            "emitted_trade_precision": None,
        }
        if name != "majority_no_trade":
            row.update(
                {
                    "filter_roc_auc": result["directional_filter"]["roc_auc"],
                    "direction_roc_auc": result["buy_vs_sell_on_actual_directional"]["roc_auc"],
                    "emitted_trade_precision": result["emitted_trade_quality"]["correct_direction_precision"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostic_conclusion(
    comparison: pd.DataFrame, separation_summary: list[dict[str, Any]]
) -> dict[str, str]:
    models = comparison[comparison["model"] != "majority_no_trade"]
    filter_auc = float(models["filter_roc_auc"].max())
    direction_auc = float(models["direction_roc_auc"].max())
    best_buy_recall = float(models["buy_recall"].max())
    best_sell_recall = float(models["sell_recall"].max())
    validation_direction = next(
        row for row in separation_summary
        if row["split"] == "validation" and row["comparison"] == "BUY_vs_SELL"
    )
    if direction_auc >= 0.6 and best_buy_recall >= 0.15 and best_sell_recall >= 0.15:
        verdict = "DIRECTIONAL_SIGNAL_PRESENT"
        answer = "NO_TRADE dominance includes a learnable filter and the saved models retain meaningful BUY/SELL discrimination."
    elif filter_auc >= 0.6 and direction_auc < 0.56:
        verdict = "FILTER_ONLY_SIGNAL"
        answer = "NO_TRADE dominance is primarily a learnable activity filter; the current target/features do not provide sufficient BUY/SELL direction signal."
    else:
        verdict = "INSUFFICIENT_DIRECTIONAL_SIGNAL"
        answer = "The saved models beat the majority baseline but do not demonstrate robust, balanced BUY/SELL learning under the current target/features."
    return {
        "verdict": verdict,
        "answer": answer,
        "evidence": (
            f"Best filter AUC={filter_auc:.4f}, best BUY-vs-SELL AUC={direction_auc:.4f}, "
            f"best validation BUY/SELL recall={best_buy_recall:.4f}/{best_sell_recall:.4f}, "
            f"validation median BUY-vs-SELL feature KS={validation_direction['median_ks']:.4f}."
        ),
    }


def render_report(report: dict[str, Any], comparison: pd.DataFrame) -> str:
    lines = [
        f"# {DIAGNOSTIC_ID}",
        "",
        "- Mode: read-only diagnostic; no fitting or tuning",
        "- TEST labels read/evaluated: false / false",
        "",
        "## Class Distribution",
        "| Split | Class | Count | Percentage |",
        "|---|---|---:|---:|",
    ]
    for split, rows in report["class_distribution"].items():
        for row in rows:
            lines.append(f"| {split.upper()} | {row['class']} | {row['count']:,} | {row['percentage']:.2f}% |")
    lines.extend(
        [
            "",
            "## Validation Model Comparison",
            "| Model | Accuracy | Balanced acc. | Macro F1 | SELL P/R | BUY P/R | NO_TRADE P/R | Predictions S/B/N | Filter AUC | Direction AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        filter_auc = "n/a" if pd.isna(row.filter_roc_auc) else f"{row.filter_roc_auc:.4f}"
        direction_auc = "n/a" if pd.isna(row.direction_roc_auc) else f"{row.direction_roc_auc:.4f}"
        lines.append(
            f"| {row.model} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.sell_precision:.4f}/{row.sell_recall:.4f} | {row.buy_precision:.4f}/{row.buy_recall:.4f} | "
            f"{row.no_trade_precision:.4f}/{row.no_trade_recall:.4f} | {row.sell_prediction_pct:.2f}%/"
            f"{row.buy_prediction_pct:.2f}%/{row.no_trade_prediction_pct:.2f}% | {filter_auc} | {direction_auc} |"
        )
    lines.extend(
        [
            "",
            "## Feature Distribution Separation",
            "| Split | Comparison | Median KS | Max KS | Features KS>=0.10 | Median robust effect |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["feature_separation_summary"]:
        lines.append(
            f"| {row['split'].upper()} | {row['comparison']} | {row['median_ks']:.4f} | "
            f"{row['max_ks']:.4f} | {row['features_ks_at_least_0_1']} | "
            f"{row['median_absolute_median_iqr_effect']:.4f} |"
        )
    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "KS and robust median effects describe marginal distribution separation; they do not establish tradable causality.",
            "",
            "## Conclusion",
            f"**{conclusion['verdict']}**: {conclusion['answer']}",
            "",
            conclusion["evidence"],
            "",
            "Do not proceed to another feature-removal experiment on the basis of NO_TRADE accuracy alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_diagnostic(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    diagnostic_report = json.loads(DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8"))
    features = list(metadata["feature_names"])
    protected = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        **{
            f"{name}_model_sha256": sha256(spec["model"])
            for name, spec in MODEL_SPECS.items()
        },
    }
    train, validation, access = load_train_validation_only(
        DATASET_PATH, metadata, [*features, "target"]
    )
    if access["rows_exposed"] != 79908 or access["test_labels_exposed"]:
        raise RuntimeError("Diagnostic violated sealed TRAIN+VALIDATION boundary")

    separation = pd.concat(
        [
            feature_separation(train, features, "train"),
            feature_separation(validation, features, "validation"),
        ],
        ignore_index=True,
    )
    separation_summary = summarize_separation(separation)
    model_results = {}
    loaded_features = {}
    for name, spec in MODEL_SPECS.items():
        model = XGBoostModel(models_dir=str(spec["model"].parent))
        model.load(spec["id"])
        loaded_features[name] = model.feature_columns
        model_results[name] = evaluate_saved_model(model, validation)
    majority = majority_evaluation(validation)
    comparison = model_comparison_rows(majority, model_results)
    conclusion = diagnostic_conclusion(comparison, separation_summary)

    after = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        **{
            f"{name}_model_sha256": sha256(spec["model"])
            for name, spec in MODEL_SPECS.items()
        },
    }
    if after != protected:
        raise RuntimeError(f"Protected artifact changed: before={protected}, after={after}")
    report = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "mode": "read-only; loaded existing models; no fitting, tuning, or target changes",
        "target_definition": metadata["target_definition"],
        "data_access": access,
        "class_distribution": {
            "train": class_distribution(train),
            "validation": class_distribution(validation),
        },
        "majority_baseline": majority,
        "saved_model_validation": model_results,
        "model_comparison": comparison.astype(object).where(
            pd.notna(comparison), None
        ).to_dict(orient="records"),
        "feature_separation_summary": separation_summary,
        "feature_families": diagnostic_report["feature_families"],
        "loaded_feature_order": loaded_features,
        "conclusion": conclusion,
        "integrity": {
            "before": protected,
            "after": after,
            "all_protected_artifacts_unchanged": after == protected,
            "test_rows_exposed": 0,
            "test_labels_read": False,
            "test_evaluated": False,
        },
    }
    output_dir.mkdir(parents=True)
    separation.to_csv(output_dir / "feature_separation.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "diagnostic_report.md").write_text(
        render_report(report, comparison), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run_diagnostic()
    print(json.dumps(result["conclusion"], indent=2))