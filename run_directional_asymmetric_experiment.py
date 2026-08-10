"""Controlled asymmetric BUY/SELL success experiment on the frozen OHLCV baseline.

Research question
-----------------
Does separating BUY and SELL into independent binary success questions extract
directional information that the frozen BUY-vs-SELL formulation failed to capture?

Target definitions (locked before training)
-------------------------------------------
The frozen V0.3 endpoint labels already encode predefined BUY and SELL
objectives on the same ATR% barriers and five-bar horizon:

  BUY_success[t]  = 1  iff  Close[t+5]/Close[t] - 1  >  +ATR%[t]
  SELL_success[t] = 1  iff  Close[t+5]/Close[t] - 1  <  -ATR%[t]

These are exactly ``target == 1`` and ``target == 0`` in the frozen dataset.

Why not invent a new path-ordered target here
---------------------------------------------
The user wording ("objective before adverse") literally describes first-passage.
That construction was already evaluated as a multiclass directional control
(``first_passage_1atr_5bar_v1``, validation BUY-vs-SELL AUC = 0.4994) and found
filter-only signal. Replacing the target *and* the formulation in one step would
confound the ablation. This experiment therefore holds the frozen endpoint events
fixed and changes only the prediction question (one-vs-rest + combination).

Conceptual distinction (predeclared)
------------------------------------
BUY_success and SELL_success are mutually exclusive but not complementary:
both are 0 on NO_TRADE rows. Therefore the asymmetric pair is not a trivial
relabeling of BUY-vs-SELL on directional rows alone. Whether that difference
yields useful information is an empirical question answered below.

TEST remains sealed. Frozen dataset and XGB baseline are never modified.
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_directional_asymmetric_buy_sell_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
DIRECTIONAL_REPORT_PATH = (
    BASELINE_DIR
    / f"{BASELINE_ID}_directional_signal_diagnostic_v1"
    / "diagnostic_report.json"
)
FIRST_PASSAGE_REPORT_PATH = (
    BASELINE_DIR
    / f"{BASELINE_ID}_first_passage_1atr_5bar_v1"
    / "experiment_report.json"
)

EXPECTED_DATASET_SHA256 = "35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634"
EXPECTED_METADATA_SHA256 = "f3a4667ffc2ca8b7dd1fddf0a44c842f0a871401b88f77c57e8e7aa25b79b042"
EXPECTED_BASELINE_MODEL_SHA256 = "6e30c834df78c6448a6987b9427e1a7ee0677eacaf15efd04e82a828d207f3b8"
FROZEN_XGB_DIRECTIONAL_AUC = 0.5129618200774144

RANDOM_SEED = 42
PERMUTATIONS = 199
N_CV_SPLITS = 5
DECISION_THRESHOLD = 0.5

# Predeclared materiality gates (do not revise after seeing results).
PROMISING_AUC = 0.54
MINIMUM_CONTROL_GAIN = 0.02
MINIMUM_CV_MEAN_AUC = 0.52
MAXIMUM_OVERFIT_GAP = 0.10
PERMUTATION_ALPHA = 0.05

PROTECTED_PATHS = {
    "dataset": DATASET_PATH,
    "metadata": METADATA_PATH,
    "baseline_model": BASELINE_MODEL_PATH,
    "baseline_evaluation": BASELINE_REPORT_PATH,
    "directional_diagnostic": DIRECTIONAL_REPORT_PATH,
}


def protected_hashes() -> dict[str, str]:
    return {f"{name}_sha256": sha256(path) for name, path in PROTECTED_PATHS.items()}


def assert_frozen_integrity(reference_report: dict[str, Any]) -> dict[str, str]:
    actual = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "baseline_model_sha256": sha256(BASELINE_MODEL_PATH),
    }
    expected = {
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "baseline_model_sha256": EXPECTED_BASELINE_MODEL_SHA256,
    }
    if actual != expected:
        raise RuntimeError(f"Frozen artifact hash mismatch: {actual} != {expected}")
    recorded = reference_report["dataset"]["sha256"]
    if recorded != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Baseline evaluation dataset hash disagrees with expected frozen hash")
    directional = json.loads(DIRECTIONAL_REPORT_PATH.read_text(encoding="utf-8"))
    control_auc = directional["saved_model_validation"]["reference_42f"][
        "buy_vs_sell_on_actual_directional"
    ]["roc_auc"]
    if control_auc != FROZEN_XGB_DIRECTIONAL_AUC:
        raise RuntimeError(f"Recorded frozen directional AUC drifted: {control_auc}")
    return actual


def attach_asymmetric_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    target = out["target"].to_numpy(dtype=int)
    out["buy_success"] = (target == 1).astype(np.int8)
    out["sell_success"] = (target == 0).astype(np.int8)
    if bool(((out["buy_success"] == 1) & (out["sell_success"] == 1)).any()):
        raise RuntimeError("BUY_success and SELL_success are not mutually exclusive")
    return out


def label_relationship_audit(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    def summarize(frame: pd.DataFrame, split: str) -> dict[str, Any]:
        buy = frame["buy_success"].to_numpy(dtype=int)
        sell = frame["sell_success"].to_numpy(dtype=int)
        both_zero = int(((buy == 0) & (sell == 0)).sum())
        both_one = int(((buy == 1) & (sell == 1)).sum())
        directional = int(((buy == 1) | (sell == 1)).sum())
        correlation = float(np.corrcoef(buy, sell)[0, 1]) if len(frame) > 1 else float("nan")
        return {
            "split": split,
            "rows": len(frame),
            "buy_success_prevalence": float(buy.mean()),
            "sell_success_prevalence": float(sell.mean()),
            "both_zero_count": both_zero,
            "both_zero_rate": float(both_zero / len(frame)),
            "both_one_count": both_one,
            "directional_count": directional,
            "label_pearson_correlation": correlation,
            "are_strict_complements": both_zero == 0 and both_one == 0,
            "are_mutually_exclusive": both_one == 0,
        }

    return {
        "interpretation": (
            "If both_zero_rate > 0, asymmetric labels are not complements of each other "
            "and therefore not a trivial BUY-vs-SELL relabeling on the full sample."
        ),
        "train": summarize(train, "train"),
        "validation": summarize(validation, "validation"),
    }


def binary_frame(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    out = frame.copy()
    out["target"] = out[label_column].astype(np.int8)
    return out


def positive_probability(model: XGBoostModel, frame: pd.DataFrame) -> np.ndarray:
    values = model.scaler.transform(frame[model.feature_columns].to_numpy())
    probabilities = model.model.predict_proba(values)
    classes = {int(label): position for position, label in enumerate(model.model.classes_)}
    if 1 not in classes:
        return np.zeros(len(frame), dtype=float)
    return probabilities[:, classes[1]]


def binary_metrics(actual: np.ndarray, score: np.ndarray, *, threshold: float = DECISION_THRESHOLD) -> dict[str, Any]:
    predicted = (score >= threshold).astype(int)
    prevalence = float(actual.mean())
    metrics: dict[str, Any] = {
        "rows": int(len(actual)),
        "positive_count": int(actual.sum()),
        "prevalence": prevalence,
        "threshold": threshold,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, score)),
    }
    if len(np.unique(actual)) < 2:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["calibration"] = None
        return metrics
    metrics["roc_auc"] = float(roc_auc_score(actual, score))
    metrics["pr_auc"] = float(average_precision_score(actual, score))
    try:
        fraction_positives, mean_predicted = calibration_curve(
            actual, score, n_bins=10, strategy="quantile"
        )
        metrics["calibration"] = [
            {
                "bin": index,
                "mean_predicted_probability": float(mean_predicted[index]),
                "observed_frequency": float(fraction_positives[index]),
            }
            for index in range(len(mean_predicted))
        ]
    except ValueError:
        metrics["calibration"] = None
    return metrics


def fit_binary_model(
    train: pd.DataFrame,
    features: list[str],
    parameters: dict[str, Any],
    label_column: str,
) -> XGBoostModel:
    frame = binary_frame(train, label_column)
    model = XGBoostModel(**parameters, models_dir=str(OUTPUT_DIR))
    model.fit_training_data(frame, features)
    if int(model.scaler.n_samples_seen_) != len(frame):
        raise RuntimeError("Scaler was not fit on exactly the supplied TRAIN rows")
    if set(np.unique(frame["target"])) - {0, 1}:
        raise RuntimeError("Binary label set must be {0, 1}")
    return model


def evaluate_binary_model(
    model: XGBoostModel,
    frame: pd.DataFrame,
    label_column: str,
) -> dict[str, Any]:
    actual = frame[label_column].to_numpy(dtype=int)
    score = positive_probability(model, frame)
    return {
        "label_column": label_column,
        **binary_metrics(actual, score),
    }


def combination_direction_metrics(
    buy_scores: np.ndarray,
    sell_scores: np.ndarray,
    multiclass_target: np.ndarray,
) -> dict[str, Any]:
    directional = multiclass_target != 2
    if directional.sum() < 2:
        return {
            "actual_directional_rows": int(directional.sum()),
            "roc_auc": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
        }
    buy = buy_scores[directional]
    sell = sell_scores[directional]
    score = buy / np.clip(buy + sell, 1e-12, None)
    actual = (multiclass_target[directional] == 1).astype(int)
    predicted = (score >= DECISION_THRESHOLD).astype(int)
    neither_dominates_rate = float(np.mean((buy_scores < DECISION_THRESHOLD) & (sell_scores < DECISION_THRESHOLD)))
    buy_dominates_rate = float(np.mean(buy_scores > sell_scores))
    return {
        "actual_directional_rows": int(directional.sum()),
        "roc_auc": float(roc_auc_score(actual, score)),
        "pr_auc": float(average_precision_score(actual, score)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "score_mean": float(score.mean()),
        "buy_dominates_rate_all_rows": buy_dominates_rate,
        "neither_dominates_rate_all_rows": neither_dominates_rate,
        "mean_buy_probability_all_rows": float(buy_scores.mean()),
        "mean_sell_probability_all_rows": float(sell_scores.mean()),
        "probability_pearson_correlation_all_rows": float(np.corrcoef(buy_scores, sell_scores)[0, 1]),
    }


def chronological_cv(
    train: pd.DataFrame,
    features: list[str],
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, (fit_indices, val_indices) in enumerate(
        TimeSeriesSplit(n_splits=N_CV_SPLITS).split(train), 1
    ):
        fit = train.iloc[fit_indices]
        fold_validation = train.iloc[val_indices]
        if fit.index[-1] >= fold_validation.index[0]:
            raise RuntimeError("Chronological ordering violated in CV fold")

        buy_model = fit_binary_model(fit, features, parameters, "buy_success")
        sell_model = fit_binary_model(fit, features, parameters, "sell_success")
        buy_metrics = evaluate_binary_model(buy_model, fold_validation, "buy_success")
        sell_metrics = evaluate_binary_model(sell_model, fold_validation, "sell_success")
        buy_scores = positive_probability(buy_model, fold_validation)
        sell_scores = positive_probability(sell_model, fold_validation)
        combo = combination_direction_metrics(
            buy_scores,
            sell_scores,
            fold_validation["target"].to_numpy(dtype=int),
        )
        rows.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(fold_validation),
                "buy_roc_auc": buy_metrics["roc_auc"],
                "buy_pr_auc": buy_metrics["pr_auc"],
                "buy_balanced_accuracy": buy_metrics["balanced_accuracy"],
                "sell_roc_auc": sell_metrics["roc_auc"],
                "sell_pr_auc": sell_metrics["pr_auc"],
                "sell_balanced_accuracy": sell_metrics["balanced_accuracy"],
                "combination_direction_roc_auc": combo["roc_auc"],
                "combination_direction_balanced_accuracy": combo["balanced_accuracy"],
                "scaler_fit_rows_buy": int(buy_model.scaler.n_samples_seen_),
                "scaler_fit_rows_sell": int(sell_model.scaler.n_samples_seen_),
            }
        )
        del buy_model, sell_model
        gc.collect()
    return rows


def permutation_null(
    actual: np.ndarray,
    score: np.ndarray,
    *,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    if len(np.unique(actual)) < 2:
        return {
            "method": "label permutation on sealed VALIDATION scores",
            "metric": metric,
            "permutations": PERMUTATIONS,
            "observed": None,
            "null_mean": None,
            "null_std": None,
            "one_sided_p_value": None,
        }
    if metric == "roc_auc":
        observed = float(roc_auc_score(actual, score))
        scorer = lambda labels: float(roc_auc_score(labels, score))
    elif metric == "pr_auc":
        observed = float(average_precision_score(actual, score))
        scorer = lambda labels: float(average_precision_score(labels, score))
    else:
        raise ValueError(metric)
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(PERMUTATIONS):
        null_values.append(scorer(rng.permutation(actual)))
    null_array = np.asarray(null_values, dtype=float)
    return {
        "method": "label permutation on sealed VALIDATION scores (model not refit)",
        "metric": metric,
        "permutations": PERMUTATIONS,
        "observed": observed,
        "null_mean": float(null_array.mean()),
        "null_std": float(null_array.std(ddof=1)),
        "one_sided_p_value": float((1 + np.sum(null_array >= observed)) / (PERMUTATIONS + 1)),
    }


def leakage_audit(
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    access: dict[str, Any],
) -> dict[str, Any]:
    forbidden = [
        column
        for column in features
        if column in {"target", "buy_success", "sell_success"}
        or column.startswith("future_")
        or column.endswith("_outcome")
    ]
    return {
        "test_rows_exposed": access.get("test_rows_exposed"),
        "test_labels_exposed": access.get("test_labels_exposed"),
        "last_exposed_timestamp": access.get("last_exposed_timestamp"),
        "feature_count": len(features),
        "forbidden_feature_columns": forbidden,
        "features_are_frozen_ohlcv_only": forbidden == [],
        "target_depends_on_future_close_only_via_precomputed_frozen_label": True,
        "asymmetric_labels_derived_only_from_frozen_target_column": True,
        "scaler_fit_scope": "TRAIN partition only for holdout models; fold TRAIN only for CV",
        "threshold_selection": (
            f"fixed predeclared threshold={DECISION_THRESHOLD}; no validation-tuned threshold search"
        ),
        "hyperparameter_selection": "copied exactly from frozen XGBoost baseline; no search",
        "train_validation_gap_ok": bool(train.index[-1] < validation.index[0]),
        "horizon": "5 M15 bars (75 minutes), inherited from v0.3_forward_atr_v1",
        "atr_barrier": "+/- 1.0 * ATR%[t] known at prediction time t",
    }


def classify_verdict(
    *,
    control_direction_auc: float,
    buy_val: dict[str, Any],
    sell_val: dict[str, Any],
    combo_val: dict[str, Any],
    cv_rows: list[dict[str, Any]],
    permutations: dict[str, Any],
) -> dict[str, Any]:
    cv = pd.DataFrame(cv_rows)
    combo_auc = combo_val["roc_auc"]
    combo_gain = None if combo_auc is None else float(combo_auc - control_direction_auc)
    cv_combo_mean = float(cv["combination_direction_roc_auc"].mean())
    cv_combo_std = float(cv["combination_direction_roc_auc"].std(ddof=0))
    buy_auc = buy_val["roc_auc"]
    sell_auc = sell_val["roc_auc"]
    buy_p = permutations["buy_roc_auc"]["one_sided_p_value"]
    sell_p = permutations["sell_roc_auc"]["one_sided_p_value"]
    combo_p = permutations["combination_direction_roc_auc"]["one_sided_p_value"]

    near_chance = (
        (buy_auc is None or buy_auc < PROMISING_AUC)
        and (sell_auc is None or sell_auc < PROMISING_AUC)
        and (combo_auc is None or combo_auc < PROMISING_AUC)
    )
    permutation_significant = any(
        p is not None and p <= PERMUTATION_ALPHA for p in (buy_p, sell_p, combo_p)
    )

    supported = (
        combo_auc is not None
        and combo_auc >= PROMISING_AUC
        and combo_gain is not None
        and combo_gain >= MINIMUM_CONTROL_GAIN
        and cv_combo_mean >= MINIMUM_CV_MEAN_AUC
        and combo_p is not None
        and combo_p <= PERMUTATION_ALPHA
    )

    if supported:
        verdict = "DIRECTIONAL_ASYMMETRY_SUPPORTED"
    elif near_chance and not permutation_significant:
        verdict = "NO_DIRECTIONAL_SIGNAL"
    elif (
        combo_auc is not None
        and combo_auc <= control_direction_auc + MINIMUM_CONTROL_GAIN
        and (buy_auc is None or buy_auc < PROMISING_AUC)
        and (sell_auc is None or sell_auc < PROMISING_AUC)
    ):
        verdict = "DIRECTIONAL_ASYMMETRY_NOT_SUPPORTED"
    else:
        # Weak numerical lifts that fail predeclared gates still do not support asymmetry.
        verdict = "DIRECTIONAL_ASYMMETRY_NOT_SUPPORTED"

    genuinely_different_problem = True  # from label audit; filled by caller evidence
    return {
        "verdict": verdict,
        "control_direction_auc": control_direction_auc,
        "combination_direction_auc": combo_auc,
        "combination_gain_vs_control": combo_gain,
        "cv_combination_direction_auc_mean": cv_combo_mean,
        "cv_combination_direction_auc_std": cv_combo_std,
        "buy_validation_roc_auc": buy_auc,
        "sell_validation_roc_auc": sell_auc,
        "permutation_p_values": {
            "buy_roc_auc": buy_p,
            "sell_roc_auc": sell_p,
            "combination_direction_roc_auc": combo_p,
        },
        "gates": {
            "promising_auc": PROMISING_AUC,
            "minimum_control_gain": MINIMUM_CONTROL_GAIN,
            "minimum_cv_mean_auc": MINIMUM_CV_MEAN_AUC,
            "permutation_alpha": PERMUTATION_ALPHA,
        },
        "genuinely_different_problem_from_label_construction": genuinely_different_problem,
        "notes": (
            "Asymmetric labels include NO_TRADE as negatives, so the learning problems "
            "differ from BUY-vs-SELL on directional rows. Empirical usefulness is gated "
            "separately from that mathematical distinction."
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    control = payload["control"]
    buy = payload["buy_model"]["validation"]
    sell = payload["sell_model"]["validation"]
    combo = payload["combination"]["validation"]
    cv = pd.DataFrame(payload["chronological_cv"])
    labels = payload["label_relationship_audit"]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- Verdict: **{decision['verdict']}**",
        f"- TEST rows/labels exposed: {payload['access']['test_rows_exposed']} / {payload['access']['test_labels_exposed']}",
        f"- Dataset SHA-256: `{payload['integrity']['dataset_sha256']}`",
        "",
        "## Research question",
        "Does separating BUY and SELL into independent success questions reveal directional",
        "information that the frozen BUY-vs-SELL formulation failed to extract from the same",
        "frozen OHLCV representation?",
        "",
        "## Exact targets (locked before training)",
        "- Horizon: 5 M15 bars",
        "- BUY_success: `Close[t+5]/Close[t]-1 > +ATR%[t]` (frozen `target == 1`)",
        "- SELL_success: `Close[t+5]/Close[t]-1 < -ATR%[t]` (frozen `target == 0`)",
        "- NO_TRADE rows are negatives for both binary models",
        "",
        "## Why this target",
        "Frozen endpoint events are held fixed so the ablation isolates formulation change.",
        "Literal first-passage path ordering was already tested as multiclass",
        f"(validation direction AUC = {payload['prior_first_passage_direction_auc']}).",
        "",
        "## Feature set",
        f"- Frozen 42 OHLCV-derived features from `{BASELINE_ID}`",
        "- No external information families added",
        "",
        "## Frozen splits",
        f"- TRAIN: {payload['splits']['train']['row_count']} rows "
        f"({payload['splits']['train']['first_timestamp']} → {payload['splits']['train']['last_timestamp']})",
        f"- VALIDATION: {payload['splits']['validation']['row_count']} rows "
        f"({payload['splits']['validation']['first_timestamp']} → {payload['splits']['validation']['last_timestamp']})",
        "- TEST: sealed / not accessed",
        "",
        "## Label relationship (are B/C different from A?)",
        f"- TRAIN both-zero rate (NO_TRADE): {labels['train']['both_zero_rate']:.4f}",
        f"- VALIDATION both-zero rate: {labels['validation']['both_zero_rate']:.4f}",
        f"- Strict complements? TRAIN={labels['train']['are_strict_complements']} "
        f"VALIDATION={labels['validation']['are_strict_complements']}",
        f"- Mutually exclusive? TRAIN={labels['train']['are_mutually_exclusive']} "
        f"VALIDATION={labels['validation']['are_mutually_exclusive']}",
        f"- Label correlation TRAIN/VAL: {labels['train']['label_pearson_correlation']:.4f} / "
        f"{labels['validation']['label_pearson_correlation']:.4f}",
        "",
        "## Models",
        "- Control: frozen multiclass XGBoost 42f (`multi:softprob`)",
        "- BUY model: binary XGBoost (`binary:logistic`) predicting BUY_success",
        "- SELL model: binary XGBoost (`binary:logistic`) predicting SELL_success",
        "- Combination: `P(BUY_success) / (P(BUY_success)+P(SELL_success))` on actual directional rows",
        "- Hyperparameters: identical to frozen baseline; no search",
        "",
        "## Results",
        "| Treatment | Val ROC-AUC | Val PR-AUC | Balanced acc. | Precision | Recall | Prevalence |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Control BUY-vs-SELL (directional rows) | {control['direction_roc_auc']:.4f} | n/a | "
            f"{control['balanced_accuracy']:.4f} | {control['precision']:.4f} | "
            f"{control['recall']:.4f} | n/a |"
        ),
        (
            f"| BUY_success | {buy['roc_auc']:.4f} | {buy['pr_auc']:.4f} | {buy['balanced_accuracy']:.4f} | "
            f"{buy['precision']:.4f} | {buy['recall']:.4f} | {buy['prevalence']:.4f} |"
        ),
        (
            f"| SELL_success | {sell['roc_auc']:.4f} | {sell['pr_auc']:.4f} | {sell['balanced_accuracy']:.4f} | "
            f"{sell['precision']:.4f} | {sell['recall']:.4f} | {sell['prevalence']:.4f} |"
        ),
        (
            f"| Combination direction score | {combo['roc_auc']:.4f} | {combo['pr_auc']:.4f} | "
            f"{combo['balanced_accuracy']:.4f} | {combo['precision']:.4f} | {combo['recall']:.4f} | n/a |"
        ),
        "",
        f"- Combination gain vs control: {decision['combination_gain_vs_control']:+.4f}",
        f"- Probability correlation (VAL): {combo['probability_pearson_correlation_all_rows']:.4f}",
        f"- Neither-dominates rate (both probs < 0.5): {combo['neither_dominates_rate_all_rows']:.4f}",
        "",
        "## Chronological CV (TRAIN only)",
        f"- BUY ROC-AUC mean/std: {cv['buy_roc_auc'].mean():.4f} / {cv['buy_roc_auc'].std(ddof=0):.4f}",
        f"- SELL ROC-AUC mean/std: {cv['sell_roc_auc'].mean():.4f} / {cv['sell_roc_auc'].std(ddof=0):.4f}",
        (
            f"- Combination direction ROC-AUC mean/std: "
            f"{cv['combination_direction_roc_auc'].mean():.4f} / "
            f"{cv['combination_direction_roc_auc'].std(ddof=0):.4f}"
        ),
        "",
        "## Null / permutation (VALIDATION label shuffles)",
        (
            f"- BUY ROC-AUC p={payload['permutations']['buy_roc_auc']['one_sided_p_value']:.4f} "
            f"(null mean {payload['permutations']['buy_roc_auc']['null_mean']:.4f})"
        ),
        (
            f"- SELL ROC-AUC p={payload['permutations']['sell_roc_auc']['one_sided_p_value']:.4f} "
            f"(null mean {payload['permutations']['sell_roc_auc']['null_mean']:.4f})"
        ),
        (
            f"- Combination direction ROC-AUC p="
            f"{payload['permutations']['combination_direction_roc_auc']['one_sided_p_value']:.4f} "
            f"(null mean {payload['permutations']['combination_direction_roc_auc']['null_mean']:.4f})"
        ),
        "",
        "## Overfitting assessment",
        (
            f"- BUY train/val ROC-AUC: {payload['buy_model']['train']['roc_auc']:.4f} / "
            f"{buy['roc_auc']:.4f} (gap {payload['buy_model']['train']['roc_auc'] - buy['roc_auc']:+.4f})"
        ),
        (
            f"- SELL train/val ROC-AUC: {payload['sell_model']['train']['roc_auc']:.4f} / "
            f"{sell['roc_auc']:.4f} (gap {payload['sell_model']['train']['roc_auc'] - sell['roc_auc']:+.4f})"
        ),
        "",
        "## Leakage audit",
        f"- Forbidden feature columns: {payload['leakage_audit']['forbidden_feature_columns']}",
        f"- TEST exposed: {payload['leakage_audit']['test_rows_exposed']}",
        f"- Train/validation chronological gap OK: {payload['leakage_audit']['train_validation_gap_ok']}",
        "",
        "## Final interpretation",
        decision["notes"],
        "",
        f"**Verdict: {decision['verdict']}**",
        "",
        "## Recommended next experiment",
        payload["recommended_next_experiment"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    ensure_output_available(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=False)

    reference_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    integrity = assert_frozen_integrity(reference_report)
    before_hashes = protected_hashes()
    parameters = model_parameters(reference_report)
    features = list(reference_report["features"]["names_in_order"])
    if len(features) != 42:
        raise RuntimeError("Frozen feature count must remain 42")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    requested_columns = ["target", *features]
    train_raw, validation_raw, access = load_train_validation_only(
        DATASET_PATH, metadata, requested_columns
    )
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("TEST leakage detected during bounded load")

    train = attach_asymmetric_labels(train_raw)
    validation = attach_asymmetric_labels(validation_raw)
    label_audit = label_relationship_audit(train, validation)

    # Control: evaluate frozen multiclass model on the same VALIDATION rows.
    control_model = XGBoostModel(models_dir=str(BASELINE_MODEL_PATH.parent))
    control_model.load(BASELINE_ID)
    control_eval = evaluate_saved_model(control_model, validation)
    control_direction = control_eval["buy_vs_sell_on_actual_directional"]
    if abs(control_direction["roc_auc"] - FROZEN_XGB_DIRECTIONAL_AUC) > 1e-12:
        raise RuntimeError("Recomputed control directional AUC does not match frozen diagnostic")

    buy_model = fit_binary_model(train, features, parameters, "buy_success")
    sell_model = fit_binary_model(train, features, parameters, "sell_success")

    buy_train = evaluate_binary_model(buy_model, train, "buy_success")
    buy_val = evaluate_binary_model(buy_model, validation, "buy_success")
    sell_train = evaluate_binary_model(sell_model, train, "sell_success")
    sell_val = evaluate_binary_model(sell_model, validation, "sell_success")

    buy_scores_val = positive_probability(buy_model, validation)
    sell_scores_val = positive_probability(sell_model, validation)
    combo_val = combination_direction_metrics(
        buy_scores_val,
        sell_scores_val,
        validation["target"].to_numpy(dtype=int),
    )
    buy_scores_train = positive_probability(buy_model, train)
    sell_scores_train = positive_probability(sell_model, train)
    combo_train = combination_direction_metrics(
        buy_scores_train,
        sell_scores_train,
        train["target"].to_numpy(dtype=int),
    )

    cv_rows = chronological_cv(train, features, parameters)

    directional_mask = validation["target"].to_numpy(dtype=int) != 2
    combo_score = buy_scores_val[directional_mask] / np.clip(
        buy_scores_val[directional_mask] + sell_scores_val[directional_mask], 1e-12, None
    )
    combo_actual = (validation["target"].to_numpy(dtype=int)[directional_mask] == 1).astype(int)
    permutations = {
        "buy_roc_auc": permutation_null(
            validation["buy_success"].to_numpy(dtype=int), buy_scores_val, metric="roc_auc", seed=RANDOM_SEED
        ),
        "sell_roc_auc": permutation_null(
            validation["sell_success"].to_numpy(dtype=int),
            sell_scores_val,
            metric="roc_auc",
            seed=RANDOM_SEED + 1,
        ),
        "combination_direction_roc_auc": permutation_null(
            combo_actual, combo_score, metric="roc_auc", seed=RANDOM_SEED + 2
        ),
    }

    decision = classify_verdict(
        control_direction_auc=control_direction["roc_auc"],
        buy_val=buy_val,
        sell_val=sell_val,
        combo_val=combo_val,
        cv_rows=cv_rows,
        permutations=permutations,
    )
    decision["genuinely_different_problem_from_label_construction"] = (
        not label_audit["validation"]["are_strict_complements"]
        and label_audit["validation"]["are_mutually_exclusive"]
    )

    prior_fp_auc = 0.49936062678464477
    if FIRST_PASSAGE_REPORT_PATH.exists():
        first_passage = json.loads(FIRST_PASSAGE_REPORT_PATH.read_text(encoding="utf-8"))
        for row in first_passage.get("comparison", []):
            if row.get("model") == "first_passage_42f":
                prior_fp_auc = float(row["direction_roc_auc"])
                break

    if decision["verdict"] == "NO_DIRECTIONAL_SIGNAL":
        recommended = (
            "Accept that asymmetric reformulation of the frozen endpoint target does not "
            "recover directional information from the current OHLCV representation. Next "
            "scientifically justified step is a new information family or a different "
            "prediction object (e.g. activity/filter deployment), not further BUY/SELL "
            "label gymnastics on the same features."
        )
    elif decision["verdict"] == "DIRECTIONAL_ASYMMETRY_NOT_SUPPORTED":
        recommended = (
            "Do not invest in asymmetric BUY/SELL model architecture on this representation. "
            "If path-ordered success remains scientifically interesting, treat it as a separate "
            "predeclared first-passage asymmetric study; otherwise move to a new information source."
        )
    else:
        recommended = (
            "Investigate calibration and simple probability comparison rules on VALIDATION only, "
            "still without touching TEST, before any strategy optimization."
        )

    after_hashes = protected_hashes()
    if after_hashes != before_hashes:
        raise RuntimeError("Protected artifacts changed during the experiment")

    buy_model_path = OUTPUT_DIR / "buy_success_xgb.joblib"
    sell_model_path = OUTPUT_DIR / "sell_success_xgb.joblib"
    joblib.dump(buy_model, buy_model_path)
    joblib.dump(sell_model, sell_model_path)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "research_question": (
            "Does separating BUY and SELL into independent success questions reveal "
            "directional information absent from the frozen BUY-vs-SELL formulation?"
        ),
        "targets": {
            "buy_success": "Close[t+5]/Close[t]-1 > +ATR%[t]  (frozen target == 1)",
            "sell_success": "Close[t+5]/Close[t]-1 < -ATR%[t]  (frozen target == 0)",
            "horizon_bars": 5,
            "atr_multiplier": 1.0,
            "source": "derived from frozen v0.3_forward_atr_v1 labels; dataset not rewritten",
        },
        "integrity": integrity,
        "protected_hashes_before": before_hashes,
        "protected_hashes_after": after_hashes,
        "access": access,
        "splits": reference_report["splits"],
        "features": features,
        "model_parameters": parameters,
        "label_relationship_audit": label_audit,
        "control": {
            "baseline_id": BASELINE_ID,
            "direction_roc_auc": control_direction["roc_auc"],
            "balanced_accuracy": control_direction["balanced_accuracy"],
            "precision": control_direction["precision"],
            "recall": control_direction["recall"],
            "accuracy": control_direction["accuracy"],
            "actual_directional_rows": control_direction["actual_directional_rows"],
            "filter_roc_auc": control_eval["directional_filter"]["roc_auc"],
            "source": "recomputed from frozen joblib on VALIDATION; matches directional diagnostic",
        },
        "buy_model": {
            "path": str(buy_model_path),
            "train": buy_train,
            "validation": buy_val,
        },
        "sell_model": {
            "path": str(sell_model_path),
            "train": sell_train,
            "validation": sell_val,
        },
        "combination": {
            "definition": "P(BUY_success)/(P(BUY_success)+P(SELL_success)) on actual BUY/SELL rows",
            "train": combo_train,
            "validation": combo_val,
        },
        "chronological_cv": cv_rows,
        "permutations": permutations,
        "leakage_audit": leakage_audit(features, train, validation, access),
        "decision": decision,
        "prior_first_passage_direction_auc": prior_fp_auc,
        "recommended_next_experiment": recommended,
        "test_policy": "not evaluated; not used for fitting, preprocessing, CV, thresholding, or selection",
    }

    report_json = OUTPUT_DIR / "experiment_report.json"
    report_md = OUTPUT_DIR / "experiment_report.md"
    cv_csv = OUTPUT_DIR / "chronological_cv.csv"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(render_report(payload), encoding="utf-8")
    pd.DataFrame(cv_rows).to_csv(cv_csv, index=False)

    # Final integrity check after writes.
    if protected_hashes() != before_hashes:
        raise RuntimeError("Protected artifacts changed after writing outputs")

    print(json.dumps({"verdict": decision["verdict"], "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
