"""Preflight tests for the frozen activity-filter confirmation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from run_ablation_25f import ensure_output_available
from run_activity_filter_confirmation_experiment import (
    HORIZON,
    PERMUTATIONS,
    RULE,
    THRESHOLDS,
    classify_confirmation,
    evaluate_threshold,
    load_confirmation_only,
    metric_differences,
)
from run_activity_filter_strategy_experiment import build_directional_rules, build_rule_outcomes


def test_manifest_is_exactly_frozen_confirmation_design():
    assert RULE == "ma_10_20"
    assert THRESHOLDS == (0.70, 0.80)
    assert HORIZON == 5
    assert PERMUTATIONS == 500


def test_confirmation_loader_exposes_only_test_without_target(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=10, freq="15min", name="Date")
    frame = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5}, index=index)
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path)
    metadata = {
        "splits": {
            "train": {"row_count": 6},
            "validation": {"row_count": 2},
            "test": {
                "row_count": 2,
                "first_timestamp": index[8].isoformat(),
                "last_timestamp": index[9].isoformat(),
            },
        }
    }

    confirmation, access = load_confirmation_only(path, metadata, ["feature"])

    assert confirmation.index.tolist() == index[8:].tolist()
    assert "target" not in confirmation
    assert access["prior_train_rows_reused_as_confirmation"] == 0
    assert access["prior_validation_rows_reused_as_confirmation"] == 0


def test_ma_rule_uses_only_exact_sma_10_and_sma_20():
    index = pd.date_range("2026-01-01", periods=3, freq="15min")
    frame = pd.DataFrame({"sma_10": [2.0, 1.0, 1.0], "sma_20": [1.0, 2.0, 1.0]}, index=index)
    ohlc = pd.DataFrame(
        {"Open": [1.0] * 3, "High": [2.0] * 3, "Low": [0.0] * 3, "Close": [1.0] * 3},
        index=index,
    )
    for column in ("sma_10", "sma_20"):
        ohlc[column] = frame[column]
    expanded = pd.DataFrame(index=index)
    expanded["sma_10"] = frame["sma_10"]
    expanded["sma_20"] = frame["sma_20"]

    directions = build_directional_rules(expanded, ohlc)["ma_10_20"]

    assert directions.tolist() == [1.0, -1.0, 0.0]


def test_confirmation_outcomes_exclude_exact_five_bar_tail():
    index = pd.date_range("2026-01-01", periods=12, freq="15min")
    close = pd.Series(np.arange(100.0, 112.0), index=index)
    ohlc = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close})
    directions = pd.Series([1.0, -1.0, 0.0, *([1.0] * 9)], index=index)

    outcomes = build_rule_outcomes(ohlc, pd.Series(2.0, index=index), directions)

    assert len(outcomes) == len(index) - 5 - 1
    assert index[2] not in outcomes.index
    assert outcomes["maximum_outcome_timestamp"].max() == index[-1]


def _metrics(trades: int, average: float, win: float, profit_factor: float) -> dict:
    return {
        "trades": trades,
        "average_return": average,
        "win_rate": win,
        "profit_factor": profit_factor,
    }


def test_confirmation_requires_both_predeclared_thresholds():
    control = _metrics(5000, 0.00001, 0.49, 1.02)
    supported = [
        {"threshold": threshold, "metrics": _metrics(3000, 0.00002, 0.50, 1.05), "permutation": {"one_sided_p_value": 0.03}}
        for threshold in THRESHOLDS
    ]

    assert classify_confirmation(control, supported)["verdict"] == "CONFIRMED_ACTIVITY_FILTER"
    supported[1]["permutation"]["one_sided_p_value"] = 0.20
    assert classify_confirmation(control, supported)["verdict"] == "FAILED_CONFIRMATION"


def test_small_confirmation_is_inconclusive():
    control = _metrics(1200, 0.0, 0.5, 1.0)
    treatments = [
        {"threshold": threshold, "metrics": _metrics(999, 0.1, 0.6, 1.2), "permutation": {"one_sided_p_value": 0.01}}
        for threshold in THRESHOLDS
    ]

    assert classify_confirmation(control, treatments)["verdict"] == "INCONCLUSIVE_CONFIRMATION"


def test_permutation_preserves_trade_count_and_is_deterministic():
    index = pd.date_range("2026-01-01", periods=20, freq="15min")
    outcomes = pd.DataFrame(
        {
            "direction": [1, -1] * 10,
            "terminal_return": np.linspace(-0.01, 0.01, 20),
            "mfe": np.linspace(0.0, 0.02, 20),
            "mae": np.linspace(-0.02, 0.0, 20),
            "reached_plus_1atr": [True, False] * 10,
            "reached_minus_1atr": [False, True] * 10,
            "first_passage": ["FAVORABLE", "ADVERSE"] * 10,
        },
        index=index,
    )
    scores = pd.Series(np.linspace(0.0, 1.0, 20), index=index)

    first = evaluate_threshold(outcomes, scores, 0.70, permutations=20)
    second = evaluate_threshold(outcomes, scores, 0.70, permutations=20)

    assert first == second
    assert first["metrics"]["trades"] == 6
    assert first["permutation"]["trade_count_preserved_each_iteration"] is True


def test_metric_differences_cover_required_metrics():
    control = {name: 1 for name in (
        "signals", "trades", "buy_trades", "sell_trades", "win_rate", "average_return",
        "median_return", "cumulative_gross_return", "profit_factor", "maximum_drawdown",
        "average_mfe", "average_mae", "reached_plus_1atr_pct", "reached_minus_1atr_pct",
    )}
    treatment = {name: 2 for name in control}

    assert set(metric_differences(treatment, control)) == set(control)
    assert all(value == 1 for value in metric_differences(treatment, control).values())


def test_confirmation_refuses_existing_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)