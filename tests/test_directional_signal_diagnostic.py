"""Focused tests for the read-only directional-signal diagnostic."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diagnose_directional_signal import (
    class_distribution,
    diagnostic_conclusion,
    ensure_output_available,
    feature_separation,
    majority_evaluation,
    model_comparison_rows,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [-2.0, -1.0, 1.0, 2.0, 0.0, 0.1, 0.2, 0.3],
            "target": [0, 0, 1, 1, 2, 2, 2, 2],
        }
    )


def test_class_distribution_accounts_for_every_row():
    distribution = class_distribution(_frame())

    assert [row["class"] for row in distribution] == ["SELL", "BUY", "NO_TRADE"]
    assert sum(row["count"] for row in distribution) == 8
    assert sum(row["percentage"] for row in distribution) == pytest.approx(100.0)


def test_majority_baseline_predicts_only_no_trade():
    metrics = majority_evaluation(_frame())

    assert metrics["balanced_accuracy"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["BUY"]["recall"] == 0
    assert metrics["per_class"]["SELL"]["recall"] == 0
    assert metrics["prediction_distribution"]["NO_TRADE"]["percentage"] == 100


def test_feature_separation_reports_all_three_pairings():
    results = feature_separation(_frame(), ["feature"], "validation")

    assert set(results["comparison"]) == {
        "BUY_vs_NO_TRADE",
        "SELL_vs_NO_TRADE",
        "BUY_vs_SELL",
    }
    assert np.isfinite(results["absolute_median_iqr_effect"]).all()
    assert (results["ks_statistic"] >= 0).all()


def test_model_comparison_keeps_majority_reference():
    majority = majority_evaluation(_frame())

    comparison = model_comparison_rows(majority, {})

    assert comparison.iloc[0]["model"] == "majority_no_trade"
    assert comparison.iloc[0]["balanced_accuracy"] == pytest.approx(1 / 3)
    assert pd.isna(comparison.iloc[0]["direction_roc_auc"])


def test_filter_without_direction_is_classified_filter_only():
    comparison = pd.DataFrame(
        [
            {
                "model": "reference_42f",
                "filter_roc_auc": 0.7,
                "direction_roc_auc": 0.53,
                "buy_recall": 0.2,
                "sell_recall": 0.2,
            }
        ]
    )
    separation = [
        {"split": "validation", "comparison": "BUY_vs_SELL", "median_ks": 0.03}
    ]

    conclusion = diagnostic_conclusion(comparison, separation)

    assert conclusion["verdict"] == "FILTER_ONLY_SIGNAL"


def test_diagnostic_refuses_existing_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)