"""Pre-training controls for directional feature enrichment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_directional_feature_enrichment import (
    NEW_FEATURES,
    build_directional_features,
    conditional_direction_metrics,
    decision,
    ensure_output_available,
    verify_feature_provenance,
)


def _ohlc(count: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2026-01-01", periods=count, freq="15min")
    close = pd.Series(100 + np.arange(count, dtype=float), index=index)
    frame = pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0, "Close": close},
        index=index,
    )
    return frame, pd.Series(2.0, index=index)


def test_manifest_is_exactly_18_unique_causal_features():
    control = [f"control_{position}" for position in range(42)]

    audit = verify_feature_provenance(control)

    assert len(NEW_FEATURES) == len(set(NEW_FEATURES)) == 18
    assert audit["total_feature_count"] == 60
    assert audit["target_derived_columns"] == []
    assert audit["future_derived_columns"] == []
    assert audit["all_features_causal"] is True


def test_feature_formulas_use_current_and_prior_rows():
    ohlc, atr = _ohlc()

    features = build_directional_features(ohlc, atr)

    row = features.iloc[100]
    assert row["return_lag_1"] == pytest.approx(ohlc["Close"].pct_change().iloc[99])
    assert row["return_20"] == pytest.approx(ohlc["Close"].iloc[100] / ohlc["Close"].iloc[80] - 1)
    assert row["atr_trend_40"] == pytest.approx(20.0)
    assert row["directional_persistence_20"] == pytest.approx(1.0)
    rolling = ohlc.iloc[51:101]
    expected_position = 2 * (
        (ohlc["Close"].iloc[100] - rolling["Low"].min())
        / (rolling["High"].max() - rolling["Low"].min())
    ) - 1
    assert row["range_position_50"] == pytest.approx(expected_position)
    assert row["candle_close_location"] == pytest.approx(0.0)
    assert row["signed_body_range"] == pytest.approx(0.25)


def test_future_row_perturbation_cannot_change_past_features():
    ohlc, atr = _ohlc()
    original = build_directional_features(ohlc, atr)
    changed = ohlc.copy()
    changed.iloc[101:, :] = [1.0, 1_000_000.0, -1_000_000.0, 500_000.0]

    modified = build_directional_features(changed, atr)

    pd.testing.assert_frame_equal(original.iloc[:101], modified.iloc[:101])


def test_longest_lookback_is_80_rows_without_backfill():
    ohlc, atr = _ohlc()

    features = build_directional_features(ohlc, atr)

    assert pd.isna(features.iloc[79]["return_80"])
    assert np.isfinite(features.iloc[80].to_numpy(dtype=float)).all()


def test_zero_range_candle_locations_are_neutral():
    ohlc, atr = _ohlc()
    ohlc.loc[:, "Open"] = ohlc["Close"]
    ohlc.loc[:, "High"] = ohlc["Close"]
    ohlc.loc[:, "Low"] = ohlc["Close"]

    features = build_directional_features(ohlc, atr)

    assert features.iloc[-1]["candle_close_location"] == 0.0
    assert features.iloc[-1]["signed_body_range"] == 0.0


class _IdentityScaler:
    def transform(self, values):
        return values


class _ProbabilityModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, values):
        return np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.6, 0.1], [0.7, 0.2, 0.1]])


class _ModelWrapper:
    feature_columns = ["x"]
    scaler = _IdentityScaler()
    model = _ProbabilityModel()


def test_conditional_metrics_ignore_no_trade_probability_for_direction():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 1, 0]})

    metrics = conditional_direction_metrics(_ModelWrapper(), frame)

    assert metrics["roc_auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["buy_recall"] == 1.0
    assert metrics["sell_recall"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_decision_requires_material_directional_discrimination():
    control = {"primary_directional": {"roc_auc": 0.4994}}
    weak = {"primary_directional": {"roc_auc": 0.519, "balanced_accuracy": 0.51, "buy_recall": 0.4, "sell_recall": 0.6}}
    strong = {"primary_directional": {"roc_auc": 0.61, "balanced_accuracy": 0.56, "buy_recall": 0.4, "sell_recall": 0.5}}

    assert decision(weak, control)["verdict"] == "INSUFFICIENT_DIRECTIONAL_INFORMATION"
    assert decision(strong, control)["verdict"] == "DIRECTIONAL_SIGNAL_RECOVERED"


def test_experiment_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)