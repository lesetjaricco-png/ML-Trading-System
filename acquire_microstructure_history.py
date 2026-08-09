"""Acquire a sealed, non-overwriting US30 M1 research dataset from MT5."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from diagnose_baseline import sha256


DATA_VERSION = "microstructure_mt5_v1"
OUTPUT_DIR = Path("data/context") / DATA_VERSION
DATA_PATH = OUTPUT_DIR / "US30_2022-05-12_2025-10-02T1215_M1_mt5.parquet"
PROVENANCE_PATH = DATA_PATH.with_suffix(".parquet.provenance.json")
MANIFEST_PATH = OUTPUT_DIR / "microstructure_manifest.json"
SYMBOL = "US30"
START_UTC = datetime(2022, 5, 12, tzinfo=timezone.utc)
END_EXCLUSIVE_UTC = datetime(2025, 10, 2, 12, 15, tzinfo=timezone.utc)
CHUNK_DAYS = 45
REQUIRED_SOURCE_COLUMNS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
)
CONTRACT_FIELDS = (
    "name",
    "description",
    "path",
    "currency_base",
    "currency_profit",
    "digits",
    "point",
    "trade_tick_size",
    "trade_tick_value",
    "trade_contract_size",
    "spread_float",
    "ticks_bookdepth",
)


def validate_m1_frame(frame: pd.DataFrame) -> dict[str, Any]:
    expected = ["Open", "High", "Low", "Close", "TickVolume", "Spread", "RealVolume"]
    if list(frame.columns) != expected:
        raise RuntimeError("M1 schema differs from the fixed MT5 research schema")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("M1 timestamps must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise RuntimeError("M1 timestamps must be unique and chronological")
    if frame.empty or frame.index[0] > pd.Timestamp("2022-05-19"):
        raise RuntimeError("M1 history does not cover the TRAIN start")
    if frame.index[-1] < pd.Timestamp("2025-10-02 12:14:00"):
        raise RuntimeError("M1 history does not cover the final validation decision")
    if frame.index[-1] >= pd.Timestamp(END_EXCLUSIVE_UTC.replace(tzinfo=None)):
        raise RuntimeError("M1 artifact contains a TEST-period timestamp")
    prices = frame[["Open", "High", "Low", "Close"]]
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise RuntimeError("M1 history contains invalid prices")
    if (frame["High"] < frame[["Open", "Close"]].max(axis=1)).any():
        raise RuntimeError("M1 High is below Open/Close")
    if (frame["Low"] > frame[["Open", "Close"]].min(axis=1)).any():
        raise RuntimeError("M1 Low is above Open/Close")
    return {
        "row_count": len(frame),
        "first_bar_open_utc": frame.index[0].isoformat(),
        "last_bar_open_utc": frame.index[-1].isoformat(),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "nonzero_tick_volume_rows": int((frame["TickVolume"] > 0).sum()),
        "nonzero_real_volume_rows": int((frame["RealVolume"] > 0).sum()),
        "positive_spread_rows": int((frame["Spread"] > 0).sum()),
    }


def _fetch_chunks() -> tuple[pd.DataFrame, int, int, dict[str, Any]]:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"MT5 symbol selection failed: {mt5.last_error()}")
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            raise RuntimeError("MT5 returned no US30 contract specification")
        contract = {field: getattr(info, field, None) for field in CONTRACT_FIELDS}
        chunks: list[pd.DataFrame] = []
        chunk_start = START_UTC
        while chunk_start < END_EXCLUSIVE_UTC:
            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), END_EXCLUSIVE_UTC)
            rates = mt5.copy_rates_range(
                SYMBOL, mt5.TIMEFRAME_M1, chunk_start, chunk_end
            )
            if rates is None or len(rates) == 0:
                raise RuntimeError(
                    f"MT5 returned no M1 rows for {chunk_start.isoformat()} to "
                    f"{chunk_end.isoformat()}: {mt5.last_error()}"
                )
            chunk = pd.DataFrame(rates)
            missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(chunk.columns))
            if missing:
                raise RuntimeError(f"MT5 M1 response is missing columns: {missing}")
            chunks.append(chunk[list(REQUIRED_SOURCE_COLUMNS)])
            chunk_start = chunk_end
        combined = pd.concat(chunks, ignore_index=True).sort_values("time")
        duplicate_count = int(combined.duplicated(subset=["time"]).sum())
        combined.drop_duplicates(subset=["time"], keep="last", inplace=True)
        end_epoch = int(END_EXCLUSIVE_UTC.timestamp())
        combined = combined.loc[combined["time"] < end_epoch].copy()
        index = pd.to_datetime(combined.pop("time"), unit="s", utc=True).dt.tz_localize(None)
        combined.columns = [
            "Open", "High", "Low", "Close", "TickVolume", "Spread", "RealVolume"
        ]
        combined.index = pd.DatetimeIndex(index, name="Date")
        return combined, len(chunks), duplicate_count, contract
    finally:
        mt5.shutdown()


def acquire() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Microstructure data directory already exists: {OUTPUT_DIR}")
    lock_path = OUTPUT_DIR.parent / f".{DATA_VERSION}.acquisition.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Microstructure acquisition is already active: {lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
        frame, chunk_count, duplicate_count, contract = _fetch_chunks()
        validation = validate_m1_frame(frame)
        OUTPUT_DIR.mkdir(parents=False)
        temporary_data = DATA_PATH.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary_data)
        os.replace(temporary_data, DATA_PATH)
        provenance = {
            "schema_version": 1,
            "source": "MetaTrader5 via IC Markets terminal",
            "instrument": SYMBOL,
            "instrument_identity": contract["description"],
            "timeframe": "M1",
            "timezone": "UTC",
            "timestamp_resolution": "1 second epoch converted to UTC-naive DatetimeIndex",
            "timestamp_semantics": "bar open",
            "availability_rule": "bar_open_timestamp + 1 minute",
            "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
            "requested_start_utc": START_UTC.isoformat(),
            "requested_end_exclusive_utc": END_EXCLUSIVE_UTC.isoformat(),
            "historical_coverage": validation,
            "schema": list(frame.columns),
            "chunk_count": chunk_count,
            "duplicate_count_before_deduplication": duplicate_count,
            "missing_data_behavior": "No synthetic fill; incomplete M15 groups are excluded.",
            "revision_policy": "Broker bar snapshot retrieved once; no revised series used.",
            "contract_specification": contract,
            "data_path": str(DATA_PATH),
            "data_sha256": sha256(DATA_PATH),
        }
        PROVENANCE_PATH.write_text(
            json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
        )
        manifest = {
            "version": DATA_VERSION,
            "m1": {
                **provenance,
                "provenance_path": str(PROVENANCE_PATH),
                "provenance_sha256": sha256(PROVENANCE_PATH),
            },
            "unavailable_families": {
                "historical_ticks": {
                    "status": "UNAVAILABLE_RELIABLE_HISTORY",
                    "reason": "MT5 copy_ticks_range returned zero rows at TRAIN, mid-sample, and validation checkpoints.",
                },
                "bid_ask_imbalance": {
                    "status": "UNAVAILABLE_RELIABLE_HISTORY",
                    "reason": "No historical tick or quote stream is available; M1 OHLCV cannot reconstruct bid/ask imbalance.",
                },
                "slippage": {
                    "status": "UNAVAILABLE_RELIABLE_HISTORY",
                    "reason": "No historical executions or order-book replay are available.",
                },
            },
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        return manifest
    except Exception:
        if OUTPUT_DIR.exists():
            for path in OUTPUT_DIR.iterdir():
                path.unlink(missing_ok=True)
            OUTPUT_DIR.rmdir()
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print(json.dumps(acquire(), indent=2, allow_nan=False))