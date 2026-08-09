"""Regression tests for the normalized-only 18-feature ablation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from diagnose_baseline import sha256
from run_ablation_18f import (
    ABLATION_25F_MODEL_PATH,
    ABLATION_25F_REPORT_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DIAGNOSTIC_REPORT_PATH,
    METADATA_PATH,
    ensure_output_available,
    resolve_normalized_features,
    signal_verdict,
    three_model_comparison,
)


EXPECTED_FEATURES = [
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


def test_exact_normalized_feature_selection_from_diagnostics():
    diagnostic = json.loads(DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    features = resolve_normalized_features(diagnostic, metadata)

    assert features == EXPECTED_FEATURES
    assert len(features) == 18


def test_feature_selection_refuses_order_mismatch():
    diagnostic = {"feature_families": {"normalized_relative": EXPECTED_FEATURES[::-1]}}
    metadata = {"feature_names": EXPECTED_FEATURES}

    with pytest.raises(ValueError, match="order differs"):
        resolve_normalized_features(diagnostic, metadata)


def test_experiment_refuses_existing_output_directory(tmp_path: Path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output_dir)


def test_three_model_comparison_includes_all_class_recalls():
    baseline = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8"))
    ablation_25f = json.loads(ABLATION_25F_REPORT_PATH.read_text(encoding="utf-8"))
    cv = pd.DataFrame(
        [{"row_type": "mean", "accuracy": 0.5, "macro_f1": 0.4}]
    )

    comparison = three_model_comparison(
        baseline, diagnostic, ablation_25f, ablation_25f["evaluation"], cv
    )

    assert {"BUY Recall", "SELL Recall", "NO_TRADE Recall"}.issubset(
        set(comparison["metric"])
    )
    assert list(comparison.columns) == [
        "metric",
        "reference_42f",
        "ablation_25f",
        "normalized_18f",
        "difference_18f_vs_42f",
        "difference_18f_vs_25f",
    ]


def test_reference_models_are_present_and_distinct():
    assert BASELINE_MODEL_PATH.is_file()
    assert ABLATION_25F_MODEL_PATH.is_file()
    assert sha256(BASELINE_MODEL_PATH) != sha256(ABLATION_25F_MODEL_PATH)


def test_weak_directional_recall_is_inconclusive():
    comparison = pd.DataFrame(
        [
            {"metric": "Validation Balanced Accuracy", "normalized_18f": 0.36},
            {"metric": "Validation Macro F1", "normalized_18f": 0.32},
            {"metric": "CV Mean Macro F1", "normalized_18f": 0.35},
            {"metric": "BUY Recall", "normalized_18f": 0.09},
            {"metric": "SELL Recall", "normalized_18f": 0.05},
        ]
    )

    verdict, _ = signal_verdict(comparison)

    assert verdict == "INCONCLUSIVE"