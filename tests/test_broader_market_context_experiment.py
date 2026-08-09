"""Leakage and provenance controls for broader-market context."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_broader_market_context_experiment import (
    CONTEXT_MANIFEST_PATH,
    aggregate_completed_bars,
    asof_features,
    classify_result,
    market_features,
    protected_hashes,
    verify_context_manifest,
    verify_feature_names,
    verify_timestamp_audits,
)


def _ohlc(periods: int, start: str = "2026-01-01 08:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="15min")
    close = pd.Series(100 + np.arange(periods, dtype=float), index=index)
    return pd.DataFrame(
        {"Open": close - 0.25, "High": close + 1, "Low": close - 1, "Close": close},
        index=index,
    )


def test_h1_aggregate_requires_all_four_completed_m15_bars():
    complete = _ohlc(8, "2026-01-01 10:00")
    incomplete = complete.drop(pd.Timestamp("2026-01-01 11:45"))

    complete_bars = aggregate_completed_bars(complete, "1h", 4)
    incomplete_bars = aggregate_completed_bars(incomplete, "1h", 4)

    assert list(complete_bars.index) == [pd.Timestamp("2026-01-01 11:00"), pd.Timestamp("2026-01-01 12:00")]
    assert list(incomplete_bars.index) == [pd.Timestamp("2026-01-01 11:00")]


def test_h4_aggregate_requires_all_sixteen_completed_m15_bars():
    complete = _ohlc(16, "2026-01-01 08:00")
    incomplete = complete.drop(pd.Timestamp("2026-01-01 11:45"))

    assert list(aggregate_completed_bars(complete, "4h", 16).index) == [pd.Timestamp("2026-01-01 12:00")]
    assert aggregate_completed_bars(incomplete, "4h", 16).empty


def test_prediction_cannot_use_h1_candle_before_its_close():
    source = pd.DataFrame({"h1_value": [7.0]}, index=[pd.Timestamp("2026-01-01 11:00")])
    predictions = pd.DatetimeIndex(["2026-01-01 10:30", "2026-01-01 11:00"])

    aligned, audit = asof_features(predictions, source, "", pd.Timedelta(hours=8))

    assert pd.isna(aligned.iloc[0]["h1_value"])
    assert aligned.iloc[1]["h1_value"] == 7.0
    assert audit.iloc[1]["source_timestamp"] == pd.Timestamp("2026-01-01 11:00")


def test_prediction_cannot_use_h4_candle_before_its_close():
    source = pd.DataFrame({"h4_value": [9.0]}, index=[pd.Timestamp("2026-01-01 12:00")])
    predictions = pd.DatetimeIndex(["2026-01-01 11:45", "2026-01-01 12:00"])

    aligned, _ = asof_features(predictions, source, "", pd.Timedelta(hours=24))

    assert pd.isna(aligned.iloc[0]["h4_value"])
    assert aligned.iloc[1]["h4_value"] == 9.0


def test_external_m15_bar_is_available_only_at_bar_close():
    raw = _ohlc(120)
    features = market_features(raw, "US500")
    source_close = raw.index[100] + pd.Timedelta(minutes=15)
    predictions = pd.DatetimeIndex([source_close - pd.Timedelta(minutes=1), source_close])

    _, audit = asof_features(predictions, features, "", pd.Timedelta(hours=4))

    assert audit.iloc[0]["source_timestamp"] < source_close
    assert audit.iloc[1]["source_timestamp"] == source_close


def test_stale_external_observation_is_not_forward_filled():
    source = pd.DataFrame({"value": [1.0]}, index=[pd.Timestamp("2026-01-01 10:00")])
    predictions = pd.DatetimeIndex(["2026-01-01 15:00"])

    aligned, audit = asof_features(predictions, source, "", pd.Timedelta(hours=4))

    assert pd.isna(aligned.iloc[0]["value"])
    assert pd.isna(audit.iloc[0]["source_timestamp"])


def test_timezone_mismatch_fails_closed():
    source = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2026-01-01 10:00"], tz="UTC"))
    predictions = pd.DatetimeIndex(["2026-01-01 10:00"])

    with pytest.raises(RuntimeError, match="Timezone mismatch"):
        asof_features(predictions, source, "", pd.Timedelta(hours=1))


def test_timestamp_resolution_is_normalized_without_changing_instants():
    source_index = pd.DatetimeIndex(np.array(["2026-01-01T10:00:00"], dtype="datetime64[ms]"))
    prediction_index = pd.DatetimeIndex(np.array(["2026-01-01T10:00:00"], dtype="datetime64[us]"))
    source = pd.DataFrame({"value": [3.0]}, index=source_index)

    aligned, audit = asof_features(prediction_index, source, "", pd.Timedelta(hours=1))

    assert aligned.iloc[0]["value"] == 3.0
    assert audit.iloc[0]["source_age_minutes"] == 0.0


def test_timestamp_audit_rejects_post_prediction_observation():
    audit = pd.DataFrame(
        {
            "prediction_timestamp": [pd.Timestamp("2026-01-01 10:00")],
            "source_timestamp": [pd.Timestamp("2026-01-01 10:01")],
            "source_age_minutes": [-1.0],
        }
    )

    with pytest.raises(RuntimeError, match="Future observation"):
        verify_timestamp_audits({"US500": audit})


@pytest.mark.parametrize("name", ["future_return_1", "target_hint", "buy_label", "trade_outcome", "lead_close"])
def test_target_or_future_derived_feature_names_fail_closed(name: str):
    with pytest.raises(RuntimeError, match="Forbidden"):
        verify_feature_names(["valid_context", name])


def test_context_manifest_hashes_and_point_in_time_metadata_match():
    manifest = json.loads(CONTEXT_MANIFEST_PATH.read_text(encoding="utf-8"))

    verification = verify_context_manifest(manifest)
    hashes = protected_hashes(manifest)

    assert verification["verified_symbols"] == ["US500", "USTEC"]
    assert verification["hashes_match"] is True
    assert all(len(value) == 64 for value in hashes.values())
    assert manifest["unavailable_families"]["VIX"]["status"] == "UNAVAILABLE_RELIABLE_HISTORY"
    assert manifest["unavailable_families"]["DXY"]["status"] == "UNAVAILABLE_RELIABLE_HISTORY"
    assert manifest["unavailable_families"]["US_TREASURY_YIELDS"]["status"] == "UNAVAILABLE_RELIABLE_HISTORY"


def test_decision_requires_validation_and_cv_support():
    weak = {
        "higher_timeframe": {"validation": {"directional": {"roc_auc": 0.519}, "activity_auc": 0.75}},
        "cross_market": {"validation": {"directional": {"roc_auc": 0.51}, "activity_auc": 0.74}},
        "combined": {"validation": {"directional": {"roc_auc": 0.50}, "activity_auc": 0.76}},
    }
    partial = {
        **weak,
        "combined": {"validation": {"directional": {"roc_auc": 0.55}, "activity_auc": 0.75}},
    }
    strong = {
        **weak,
        "combined": {"validation": {"directional": {"roc_auc": 0.61}, "activity_auc": 0.75}},
    }
    weak_cv = pd.DataFrame({"treatment": [name for name in weak for _ in range(5)], "direction_auc": [0.505] * 15})
    partial_cv = weak_cv.copy(); partial_cv.loc[partial_cv["treatment"] == "combined", "direction_auc"] = 0.53
    strong_cv = weak_cv.copy(); strong_cv.loc[strong_cv["treatment"] == "combined", "direction_auc"] = 0.57

    assert classify_result(weak, weak_cv)["verdict"] == "FILTER_ONLY_SIGNAL_CONFIRMED"
    assert classify_result(partial, partial_cv)["verdict"] == "PARTIAL_DIRECTIONAL_SIGNAL"
    assert classify_result(strong, strong_cv)["verdict"] == "DIRECTIONAL_SIGNAL_RECOVERED"