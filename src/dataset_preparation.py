"""Build and persist auditable supervised datasets from validated market data."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.instruments import resolve_instrument_spec

FEATURE_GENERATION_VERSION = "us30_m15_features_v1"
TARGET_GENERATION_VERSION = "v0.3_forward_atr_v1"


def feature_engineer_from_config(cfg: dict[str, Any]) -> FeatureEngineer:
    """Construct the existing feature/target implementation from configuration."""
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]
    target_cfg = cfg["target"]
    instrument_name = data_cfg["ticker"]
    instrument_spec = resolve_instrument_spec(instrument_name, fallback_point_size=0.01)
    return FeatureEngineer(
        rsi_period=feature_cfg["rsi_period"],
        macd_fast=feature_cfg["macd_fast"],
        macd_slow=feature_cfg["macd_slow"],
        macd_signal=feature_cfg["macd_signal"],
        bb_period=feature_cfg["bb_period"],
        bb_std=feature_cfg["bb_std"],
        sma_periods=feature_cfg["sma_periods"],
        ema_periods=feature_cfg["ema_periods"],
        atr_period=feature_cfg["atr_period"],
        volume_sma_period=feature_cfg["volume_sma_period"],
        timeframe=data_cfg["interval"],
        take_profit_points=target_cfg["take_profit_points"],
        stop_loss_points=target_cfg["stop_loss_points"],
        max_bars=target_cfg["max_bars"],
        same_bar_rule=target_cfg["same_bar_rule"],
        unresolved_policy=target_cfg["unresolved_policy"],
        instrument_config=cfg.get("instruments", {}),
        instrument_spec=instrument_spec,
        target_mode=cfg["experiment"]["target_mode"],
        forward_horizon=target_cfg["forward_horizon"],
        atr_threshold_multiplier=target_cfg["atr_threshold_multiplier"],
    )


def prepare_ml_dataset(
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    feature_engineer: FeatureEngineer | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform validated MT5 OHLCV into an auditable chronological ML dataset."""
    raw_provenance = dict(raw_df.attrs.get("provenance", {}))
    _validate_raw_provenance(raw_provenance, cfg)

    raw_snapshot = raw_df.copy(deep=True)
    engineer = feature_engineer or feature_engineer_from_config(cfg)
    processed = engineer.transform(raw_df, instrument_name=cfg["data"]["ticker"])
    pd.testing.assert_frame_equal(raw_df, raw_snapshot, check_flags=False)

    feature_names = list(engineer.feature_columns)
    _validate_processed_dataset(processed, feature_names)
    split_metadata = _chronological_split_metadata(processed, cfg)
    target_counts = processed["target"].astype(int).value_counts().sort_index()

    metadata: dict[str, Any] = {
        "source": raw_provenance["source"],
        "symbol": cfg["data"]["ticker"],
        "timeframe": raw_provenance.get("mt5_timeframe", cfg["data"]["interval"]),
        "research_start": cfg["data"]["start_date"],
        "requested_end": raw_provenance["requested_end"],
        "fallback_used": False,
        "broker": raw_provenance.get("broker"),
        "server": raw_provenance.get("server"),
        "feature_generation_version": FEATURE_GENERATION_VERSION,
        "target_generation_version": TARGET_GENERATION_VERSION,
        "target_mode": engineer.target_mode,
        "target_definition": (
            "At candle close t, compare Close[t+h]/Close[t]-1 with +/- "
            "atr_threshold_multiplier * ATR[t]/Close[t]. "
            "1=BUY, 0=SELL, 2=NO_TRADE; a complete h-bar horizon is required."
        ),
        "horizon_bars": engineer.forward_horizon,
        "atr_threshold_multiplier": engineer.atr_threshold_multiplier,
        "take_profit_points": engineer.take_profit_points,
        "stop_loss_points": engineer.stop_loss_points,
        "target_uses_tp_sl": False,
        "same_bar_rule": "not_applicable_to_forward_close_target",
        "raw_row_count": int(len(raw_df)),
        "processed_row_count": int(len(processed)),
        "rows_removed": int(len(raw_df) - len(processed)),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_class_distribution": {
            str(int(label)): int(count) for label, count in target_counts.items()
        },
        "first_timestamp": pd.Timestamp(processed.index.min()).isoformat(),
        "last_timestamp": pd.Timestamp(processed.index.max()).isoformat(),
        "raw_provenance": raw_provenance,
        "splits": split_metadata,
    }
    processed.attrs["dataset_metadata"] = metadata
    return processed, metadata


def save_processed_dataset(
    processed: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    output_dir: str | Path = "data/processed",
) -> tuple[Path, Path]:
    """Save processed rows and an adjacent JSON metadata document."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    symbol = str(metadata["symbol"])
    timeframe = str(metadata["timeframe"])
    start = str(metadata["research_start"])
    end = str(metadata["requested_end"])
    stem = f"{symbol}_{start}_{end}_{timeframe}_{TARGET_GENERATION_VERSION}"
    parquet_path = output_path / f"{stem}.parquet"
    metadata_path = output_path / f"{stem}.metadata.json"

    persisted_metadata = dict(metadata)
    persisted_metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    persisted_metadata["dataset_path"] = str(parquet_path)

    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    processed.to_parquet(temporary_parquet)
    with open(temporary_metadata, "w", encoding="utf-8") as handle:
        json.dump(persisted_metadata, handle, indent=2)
    os.replace(temporary_parquet, parquet_path)
    os.replace(temporary_metadata, metadata_path)
    return parquet_path, metadata_path


def _validate_raw_provenance(provenance: dict[str, Any], cfg: dict[str, Any]) -> None:
    if str(provenance.get("source", "")).lower() != "mt5":
        raise ValueError("Processed research datasets require source=mt5 provenance.")
    if bool(provenance.get("fallback_used")):
        raise ValueError("Processed research datasets reject fallback data.")
    if provenance.get("symbol") != cfg["data"]["ticker"]:
        raise ValueError("Raw provenance symbol does not match configuration.")
    if provenance.get("timeframe") != cfg["data"]["interval"]:
        raise ValueError("Raw provenance timeframe does not match configuration.")
    if provenance.get("requested_start") != cfg["data"]["start_date"]:
        raise ValueError("Raw provenance start does not match the research boundary.")


def _validate_processed_dataset(processed: pd.DataFrame, feature_names: list[str]) -> None:
    if processed.empty:
        raise ValueError("Processed dataset is empty.")
    if not processed.index.is_monotonic_increasing or not processed.index.is_unique:
        raise ValueError("Processed timestamps must be unique and chronological.")
    invalid_names = [
        name
        for name in feature_names
        if name == "target" or name.startswith("future_") or name.endswith("_outcome")
    ]
    if invalid_names:
        raise ValueError(f"Target/future columns cannot be features: {invalid_names}")
    feature_frame = processed[feature_names]
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in feature_frame.dtypes):
        raise ValueError("All model features must be numeric.")
    if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain NaN or infinite values.")
    if not set(processed["target"].astype(int).unique()).issubset({0, 1, 2}):
        raise ValueError("V0.3 target contains unsupported class labels.")


def _chronological_split_metadata(
    processed: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    row_count = len(processed)
    test_size = float(cfg["model"]["test_size"])
    validation_size = float(cfg["model"]["validation_size"])
    test_index = int(row_count * (1 - test_size))
    validation_index = int(test_index * (1 - validation_size))
    ranges = {
        "train": processed.iloc[:validation_index],
        "validation": processed.iloc[validation_index:test_index],
        "test": processed.iloc[test_index:],
    }
    if any(frame.empty for frame in ranges.values()):
        raise ValueError("Chronological split produced an empty partition.")
    return {
        name: {
            "row_count": int(len(frame)),
            "first_timestamp": pd.Timestamp(frame.index.min()).isoformat(),
            "last_timestamp": pd.Timestamp(frame.index.max()).isoformat(),
        }
        for name, frame in ranges.items()
    }
