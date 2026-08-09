from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.dataset_preparation import prepare_ml_dataset, save_processed_dataset
from src.feature_engineering import FeatureEngineer


def _config() -> dict:
    return {
        "data": {
            "ticker": "US30",
            "start_date": "2022-05-12",
            "end_date": "2024-01-01",
            "interval": "15m",
            "source": "mt5",
        },
        "features": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2,
            "sma_periods": [10, 20, 50, 200],
            "ema_periods": [12, 26],
            "atr_period": 14,
            "volume_sma_period": 20,
        },
        "model": {"test_size": 0.2, "validation_size": 0.1},
        "experiment": {"target_mode": "v0.3_forward_atr"},
        "target": {
            "take_profit_points": 100,
            "stop_loss_points": 20,
            "max_bars": 40,
            "same_bar_rule": "conservative_sl",
            "unresolved_policy": "drop",
            "forward_horizon": 5,
            "atr_threshold_multiplier": 1.0,
        },
        "instruments": {"US30": {"take_profit_points": 100, "stop_loss_points": 20}},
    }


def _raw_frame() -> pd.DataFrame:
    rows = 500
    index = pd.date_range("2022-05-12", periods=rows, freq="15min")
    close = 30_000 + np.arange(rows) * 0.5 + np.sin(np.arange(rows) / 5) * 10
    frame = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 1000 + (np.arange(rows) % 50),
        },
        index=index,
    )
    frame.index.name = "Date"
    frame.attrs["provenance"] = {
        "source": "mt5",
        "symbol": "US30",
        "timeframe": "15m",
        "mt5_timeframe": "M15",
        "requested_start": "2022-05-12",
        "requested_end": "2024-01-01",
        "actual_first_timestamp": index[0].isoformat(),
        "actual_last_timestamp": index[-1].isoformat(),
        "row_count": rows,
        "chunk_count": 1,
        "duplicate_count": 0,
        "fallback_used": False,
        "broker": "Raw Trading Ltd",
        "server": "ICMarketsSC-Demo",
    }
    return frame


def test_prepare_dataset_is_numeric_chronological_and_preserves_raw():
    raw = _raw_frame()
    snapshot = raw.copy(deep=True)

    processed, metadata = prepare_ml_dataset(raw, _config())

    pd.testing.assert_frame_equal(raw, snapshot)
    assert processed.index.is_monotonic_increasing
    assert processed.index.is_unique
    assert np.isfinite(processed[metadata["feature_names"]].to_numpy()).all()
    assert all(
        pd.api.types.is_numeric_dtype(processed[column])
        for column in metadata["feature_names"]
    )
    assert "target" not in metadata["feature_names"]
    assert not any(column.startswith("future_") for column in metadata["feature_names"])
    assert metadata["source"] == "mt5"
    assert metadata["fallback_used"] is False
    assert metadata["feature_count"] == len(metadata["feature_names"])


def test_prepare_dataset_records_ordered_chronological_splits():
    processed, metadata = prepare_ml_dataset(_raw_frame(), _config())
    splits = metadata["splits"]

    assert splits["train"]["last_timestamp"] < splits["validation"]["first_timestamp"]
    assert splits["validation"]["last_timestamp"] < splits["test"]["first_timestamp"]
    assert sum(part["row_count"] for part in splits.values()) == len(processed)


def test_prepare_dataset_rejects_fallback_provenance():
    raw = _raw_frame()
    raw.attrs["provenance"]["fallback_used"] = True

    with pytest.raises(ValueError, match="reject fallback"):
        prepare_ml_dataset(raw, _config())


def test_save_processed_dataset_writes_separate_artifacts(tmp_path):
    raw = _raw_frame()
    processed, metadata = prepare_ml_dataset(raw, _config())

    parquet_path, metadata_path = save_processed_dataset(
        processed, metadata, output_dir=tmp_path
    )

    assert parquet_path.exists()
    assert metadata_path.exists()
    assert len(pd.read_parquet(parquet_path)) == len(processed)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["feature_names"] == metadata["feature_names"]
    assert persisted["target_class_distribution"] == metadata["target_class_distribution"]


def test_v03_target_buy_sell_neutral_and_incomplete_horizon():
    horizon = 2
    index = pd.date_range("2024-01-01", periods=9, freq="15min")
    close = [100.0, 100.0, 102.0, 100.0, 98.0, 100.0, 98.5, 100.0, 100.0]
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": np.array(close) + 1.0,
            "Low": np.array(close) - 1.0,
            "Close": close,
            "Volume": [1000] * len(close),
            "atr_pct": [0.01] * len(close),
        },
        index=index,
    )
    engineer = FeatureEngineer(
        target_mode="v0.3_forward_atr",
        forward_horizon=horizon,
        atr_threshold_multiplier=1.0,
    )

    target = pd.Series(engineer._build_v03_forward_atr_target_series(frame), index=index)

    assert target.iloc[0] == 1
    assert target.iloc[2] == 0
    assert target.iloc[4] == 2
    assert target.iloc[-horizon:].isna().all()
