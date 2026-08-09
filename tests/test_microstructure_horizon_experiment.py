"""Preflight controls for the M1 and horizon directional experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acquire_microstructure_history import END_EXCLUSIVE_UTC, validate_m1_frame
from run_ablation_25f import ensure_output_available
from run_first_passage_target_experiment import INCOMPLETE_LABEL, build_first_passage_target
from run_microstructure_horizon_experiment import (
    DECISION_THRESHOLDS,
    M1_MANIFEST_PATH,
    M1_PRICE_FEATURES,
    SPREAD_FEATURES,
    build_m1_features,
    classify_candidate,
    verify_m1_manifest,
)


def _m1(start: str, periods: int, *, seconds: int = 0) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min") + pd.Timedelta(seconds=seconds)
    close = 100 + np.arange(periods, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "Open": close - 0.02,
            "High": close + 0.05,
            "Low": close - 0.05,
            "Close": close,
            "TickVolume": np.arange(periods) + 10,
            "Spread": np.arange(periods) % 3 + 50,
            "RealVolume": np.zeros(periods),
        },
        index=index,
    )


def test_m1_validation_requires_sealed_coverage_and_records_native_fields():
    frame = pd.concat(
        [
            _m1("2022-05-12 01:00", 1),
            _m1("2025-10-02 12:14", 1),
        ]
    )

    summary = validate_m1_frame(frame)

    assert summary["first_bar_open_utc"] == "2022-05-12T01:00:00"
    assert summary["last_bar_open_utc"] == "2025-10-02T12:14:00"
    assert summary["nonzero_tick_volume_rows"] == 2
    assert summary["nonzero_real_volume_rows"] == 0
    assert summary["positive_spread_rows"] == 2


def test_m1_validation_rejects_test_period_timestamp():
    frame = pd.concat(
        [
            _m1("2022-05-12 01:00", 1),
            _m1(pd.Timestamp(END_EXCLUSIVE_UTC).tz_localize(None).isoformat(), 1),
        ]
    )

    with pytest.raises(RuntimeError, match="TEST-period"):
        validate_m1_frame(frame)


def test_only_complete_m1_blocks_are_emitted_and_timestamped_at_close():
    complete = _m1("2026-01-01 10:00", 21 * 15)
    incomplete = _m1("2026-01-01 15:15", 14)
    m1 = pd.concat([complete, incomplete])
    predictions = pd.DatetimeIndex(["2026-01-01 15:15", "2026-01-01 15:30"])
    atr = pd.Series([2.0, 2.0], index=pd.DatetimeIndex(["2026-01-01 15:00", "2026-01-01 15:15"]))

    features, audit = build_m1_features(m1, predictions, atr)

    assert features.loc[predictions[0], M1_PRICE_FEATURES].notna().all()
    assert features.loc[predictions[1]].isna().all()
    assert audit.loc[predictions[0], "maximum_source_timestamp"] == predictions[0]
    assert audit.loc[predictions[0], "source_not_after_prediction"]
    assert list(features.columns) == [*M1_PRICE_FEATURES, *SPREAD_FEATURES]


def test_m1_range_atr_uses_actual_block_range():
    m1 = _m1("2026-01-01 10:00", 15)
    prediction = pd.DatetimeIndex(["2026-01-01 10:15"])
    atr = pd.Series([2.0], index=pd.DatetimeIndex(["2026-01-01 10:00"]))

    features, _ = build_m1_features(m1, prediction, atr)
    expected_range = (m1["High"].max() - m1["Low"].min()) / 2.0

    assert features.iloc[0]["m1_range_atr_ratio"] == pytest.approx(expected_range)


def test_post_prediction_m1_close_fails_closed():
    m1 = _m1("2026-01-01 10:00", 15, seconds=30)
    prediction = pd.DatetimeIndex(["2026-01-01 10:15"])
    atr = pd.Series([2.0], index=pd.DatetimeIndex(["2026-01-01 10:00"]))

    with pytest.raises(RuntimeError, match="post-prediction"):
        build_m1_features(m1, prediction, atr)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 5, 8])
def test_each_horizon_excludes_exact_unobservable_tail(horizon: int):
    index = pd.date_range("2026-01-01", periods=20, freq="15min")
    close = pd.Series(np.full(len(index), 100.0), index=index)
    ohlc = pd.DataFrame(
        {"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close},
        index=index,
    )
    atr = pd.Series(np.full(len(index), 1.0), index=index)

    target = build_first_passage_target(ohlc, atr, horizon=horizon)

    assert int((target["first_passage_target"] == INCOMPLETE_LABEL).sum()) == horizon


def test_signal_classification_requires_validation_and_cv_confirmation():
    assert classify_candidate(0.519, 0.519, 0.01) == "NO_DIRECTIONAL_SIGNAL"
    assert classify_candidate(0.53, 0.51, 0.01) == "WEAK_UNCONFIRMED_SIGNAL"
    assert classify_candidate(0.54, 0.53, 0.02) == "PROMISING_DIRECTIONAL_SIGNAL"
    assert classify_candidate(0.58, 0.56, 0.03) == "ROBUST_DIRECTIONAL_SIGNAL"
    assert DECISION_THRESHOLDS["robust_max_cv_std"] == 0.04


def test_acquired_m1_manifest_and_hashes_are_verified():
    manifest = json.loads(M1_MANIFEST_PATH.read_text(encoding="utf-8"))

    verification = verify_m1_manifest(manifest)

    assert verification["data_hash_verified"] is True
    assert verification["provenance_hash_verified"] is True
    assert verification["historical_ticks"] == "UNAVAILABLE_RELIABLE_HISTORY"
    assert manifest["m1"]["historical_coverage"]["row_count"] == 1_200_410
    assert manifest["m1"]["historical_coverage"]["nonzero_real_volume_rows"] == 0


def test_experiment_refuses_existing_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)