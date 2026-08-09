"""Tests for the reproducible V0.3 baseline runner."""

import numpy as np
import pandas as pd

from train_baseline import _checks_pass, chronological_partitions, evaluation_metrics


def test_chronological_partitions_preserve_recorded_boundaries():
    index = pd.date_range("2026-01-01", periods=10, freq="15min")
    dataset = pd.DataFrame({"feature": range(10), "target": [0, 1, 2, 0, 1] * 2}, index=index)
    metadata = {
        "splits": {
            "train": {
                "row_count": 6,
                "first_timestamp": index[0].isoformat(),
                "last_timestamp": index[5].isoformat(),
            },
            "validation": {
                "row_count": 2,
                "first_timestamp": index[6].isoformat(),
                "last_timestamp": index[7].isoformat(),
            },
            "test": {
                "row_count": 2,
                "first_timestamp": index[8].isoformat(),
                "last_timestamp": index[9].isoformat(),
            },
        }
    }

    train, validation = chronological_partitions(dataset, metadata)

    pd.testing.assert_frame_equal(train, dataset.iloc[:6])
    pd.testing.assert_frame_equal(validation, dataset.iloc[6:8])
    assert validation.index[-1] < dataset.index[8]


def test_evaluation_metrics_report_directional_and_class_distribution():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 2, 1, 0, 2, 2])

    metrics = evaluation_metrics(y_true, y_pred)

    assert metrics["confusion_matrix"] == [[1, 0, 1], [1, 1, 0], [0, 0, 2]]
    assert metrics["directional"]["accuracy_on_actual_buy_sell_rows"] == 0.5
    assert metrics["prediction_distribution"]["NO_TRADE"]["count"] == 3
    assert metrics["directional"]["predicted_buy_sell_percentage"] == 50.0


def test_baseline_checks_require_test_to_remain_unevaluated():
    checks = {
        "artifact_reload_predictions_identical": True,
        "scaler_fit_row_count": 71917,
        "scaler_fit_matches_train_row_count": True,
        "test_evaluated": False,
    }

    assert _checks_pass(checks)
    checks["test_evaluated"] = True
    assert not _checks_pass(checks)