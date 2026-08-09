"""Preflight controls for the frozen-model activity-filter strategy test."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_ablation_25f import ensure_output_available
from run_activity_filter_strategy_experiment import (
    RULES,
    activity_probability,
    build_directional_rules,
    build_rule_outcomes,
    classify_result,
    evaluate_filter_controls,
    strategy_metrics,
)


class _IdentityScaler:
    def transform(self, values):
        return values


class _ProbabilityModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, values):
        return np.asarray([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])


class _FrozenWrapper:
    feature_columns = ["feature"]
    scaler = _IdentityScaler()
    model = _ProbabilityModel()


def _market(rows: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=rows, freq="15min")
    close = pd.Series(100 + np.arange(rows, dtype=float), index=index)
    ohlc = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.5,
            "Low": close - 1.0,
            "Close": close,
        },
        index=index,
    )
    frame = pd.DataFrame(
        {
            "sma_10": close - 1,
            "sma_20": close - 2,
            "atr": np.full(rows, 2.0),
        },
        index=index,
    )
    return frame, ohlc


def _outcomes(rows: int = 20) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="15min")
    returns = np.linspace(-0.01, 0.02, rows)
    return pd.DataFrame(
        {
            "direction": np.where(np.arange(rows) % 2, 1, -1),
            "terminal_return": returns,
            "mfe": np.abs(returns) + 0.005,
            "mae": -np.abs(returns) / 2,
            "reached_plus_1atr": returns > 0.005,
            "reached_minus_1atr": returns < -0.005,
            "first_passage": np.where(returns > 0.005, "FAVORABLE", "UNRESOLVED"),
            "maximum_outcome_timestamp": index + pd.Timedelta(minutes=75),
        },
        index=index,
    )


def test_activity_probability_sums_buy_and_sell_only():
    frame = pd.DataFrame({"feature": [1.0, 2.0]})

    result = activity_probability(_FrozenWrapper(), frame)

    np.testing.assert_allclose(result.to_numpy(), [0.5, 0.5])


def test_directional_rules_are_fixed_and_do_not_use_model_scores():
    frame, ohlc = _market()

    rules = build_directional_rules(frame, ohlc)

    assert list(rules.columns) == list(RULES)
    assert rules.iloc[-1]["momentum_3"] == 1
    assert rules.iloc[-1]["ma_10_20"] == 1
    assert rules.iloc[-1]["completed_bar_direction"] == 1
    assert "activity_probability" not in rules.columns


def test_outcomes_exclude_exact_horizon_tail_and_follow_fixed_direction():
    frame, ohlc = _market(12)
    directions = pd.Series(1, index=ohlc.index)
    directions.iloc[0] = np.nan

    outcomes = build_rule_outcomes(ohlc, frame["atr"], directions, horizon=5)

    assert len(outcomes) == 6
    assert outcomes.index[0] == ohlc.index[1]
    assert outcomes.index[-1] == ohlc.index[-6]
    assert (outcomes["terminal_return"] > 0).all()
    assert (outcomes["maximum_outcome_timestamp"] > outcomes.index).all()


def test_frequency_matched_random_filter_preserves_trade_count():
    outcomes = _outcomes()
    scores = pd.Series(np.linspace(0, 1, len(outcomes)), index=outcomes.index)

    result = evaluate_filter_controls(
        outcomes,
        scores,
        split="validation",
        rule="momentum_3",
        threshold=0.70,
        permutations=25,
    )

    assert result["ml_filter"]["trades"] == result["frequency_matched_random_filter"]["trades"]
    assert result["permutation"]["trade_count_preserved_each_iteration"] is True
    assert result["permutation"]["iterations"] == 25


def test_strategy_metrics_report_required_gross_outcomes():
    outcomes = _outcomes()
    metrics = strategy_metrics(outcomes, pd.Series(True, index=outcomes.index))

    for name in (
        "win_rate",
        "average_return",
        "median_return",
        "cumulative_gross_return",
        "profit_factor",
        "maximum_drawdown",
        "average_mfe",
        "average_mae",
        "reached_plus_1atr_pct",
        "reached_minus_1atr_pct",
    ):
        assert name in metrics
    assert metrics["maximum_drawdown"] >= 0


def _decision_fixture(*, supported_validation: int, supported_train: int, unfiltered_edge: bool = False):
    results = {"train": {}, "validation": {}}
    for split, supported_count in (("train", supported_train), ("validation", supported_validation)):
        for rule in RULES:
            control_return = 0.001 if unfiltered_edge else 0.0
            thresholds = []
            for position, threshold in enumerate((0.5, 0.6, 0.7, 0.8)):
                supported = rule == "momentum_3" and position < supported_count
                thresholds.append(
                    {
                        "threshold": threshold,
                        "ml_filter": {"average_return": 0.002 if supported else -0.001, "win_rate": 0.6 if supported else 0.4},
                        "frequency_matched_random_filter": {"average_return": 0.0005, "win_rate": 0.5},
                        "permutation": {"one_sided_p_value": 0.01 if supported else 0.5},
                    }
                )
            results[split][rule] = {
                "unfiltered": {"average_return": control_return, "win_rate": 0.5, "profit_factor": 1.1 if unfiltered_edge else 0.9},
                "thresholds": thresholds,
            }
    return results


def test_decision_requires_multiple_thresholds_and_train_confirmation():
    robust = classify_result(_decision_fixture(supported_validation=3, supported_train=2))
    weak = classify_result(_decision_fixture(supported_validation=1, supported_train=0))
    none = classify_result(_decision_fixture(supported_validation=0, supported_train=0))
    directional = classify_result(_decision_fixture(supported_validation=0, supported_train=0, unfiltered_edge=True))

    assert robust["verdict"] == "ACTIVITY_FILTER_HAS_ECONOMIC_VALUE"
    assert weak["verdict"] == "ACTIVITY_FILTER_WEAK_EVIDENCE"
    assert none["verdict"] == "ACTIVITY_FILTER_NO_ECONOMIC_VALUE"
    assert directional["verdict"] == "DIRECTIONAL_RULE_DOMINATES"


def test_experiment_refuses_existing_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)