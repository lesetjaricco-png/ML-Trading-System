"""Causal alignment and normalized external-information contract tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.external_information import (
    SCHEMA_COLUMNS,
    atomic_write_parquet,
    causal_asof_join,
    normalize_observations,
    sha256_file,
    validate_feature_names,
)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "official",
                "dataset_version": "v1",
                "instrument_or_event_id": "CPI",
                "field": "actual",
                "value": 3.0,
                "source_time_utc": "2025-01-01T13:30:00Z",
                "available_time_utc": "2025-01-01T13:30:00Z",
                "received_time_utc": "2025-01-01T13:30:01Z",
                "revision_id": "initial",
                "quality_flags": "",
                "raw_batch_sha256": "a" * 64,
            },
            {
                "source": "official",
                "dataset_version": "v1",
                "instrument_or_event_id": "CPI",
                "field": "actual",
                "value": 3.1,
                "source_time_utc": "2025-01-01T13:30:00Z",
                "available_time_utc": "2025-02-01T13:30:00Z",
                "received_time_utc": "2025-02-01T13:30:01Z",
                "revision_id": "revision_1",
                "quality_flags": "revised",
                "raw_batch_sha256": "b" * 64,
            },
        ],
        columns=SCHEMA_COLUMNS,
    )


def test_future_release_and_revision_are_unavailable_before_publication():
    predictions = pd.DatetimeIndex(
        ["2025-01-01T13:29:00Z", "2025-01-01T13:30:00Z", "2025-01-20T00:00:00Z", "2025-02-01T13:30:00Z"]
    )

    features, audit = causal_asof_join(
        predictions,
        _observations(),
        feature_name="macro_actual",
        maximum_age=pd.Timedelta(days=60),
    )

    assert pd.isna(features.iloc[0, 0])
    assert features.iloc[1:, 0].tolist() == [3.0, 3.0, 3.1]
    assert (audit.dropna(subset=["available_time_utc"])["available_time_utc"] <= audit.dropna(subset=["available_time_utc"]).index).all()


def test_stale_observation_is_rejected():
    features, _ = causal_asof_join(
        pd.DatetimeIndex(["2025-01-03T13:30:01Z"]),
        _observations().iloc[:1],
        feature_name="macro_actual",
        maximum_age=pd.Timedelta(days=1),
    )

    assert pd.isna(features.iloc[0, 0])


def test_timezone_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="timezone-aware UTC"):
        causal_asof_join(
            pd.DatetimeIndex(["2025-01-01 13:30"]),
            _observations(),
            feature_name="macro_actual",
            maximum_age=pd.Timedelta(days=1),
        )


@pytest.mark.parametrize("name", ["target_hint", "future_close", "buy_label", "trade_outcome", "lead_price"])
def test_target_or_future_names_are_rejected(name: str):
    with pytest.raises(RuntimeError, match="Forbidden"):
        validate_feature_names([name])


def test_normalization_rejects_availability_before_source_time():
    frame = _observations().iloc[:1].copy()
    frame.loc[0, "available_time_utc"] = "2025-01-01T13:29:59Z"

    with pytest.raises(ValueError, match="before its source time"):
        normalize_observations(frame)


def test_atomic_parquet_refuses_overwrite_and_hash_is_deterministic(tmp_path):
    path = tmp_path / "observations.parquet"
    normalized = normalize_observations(_observations())

    atomic_write_parquet(normalized, path)
    first_hash = sha256_file(path)

    assert len(first_hash) == 64
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        atomic_write_parquet(normalized, path)
    assert sha256_file(path) == first_hash
