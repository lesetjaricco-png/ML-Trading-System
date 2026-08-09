"""Acquire reproducible point-in-time cross-market context from MT5."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from diagnose_baseline import sha256
from src.data_ingestion import DataIngestion


CONTEXT_DATA_VERSION = "directional_context_mt5_v1"
CONTEXT_DIR = Path("data/context") / CONTEXT_DATA_VERSION
REQUESTED_START = "2022-05-12"
REQUESTED_END = "2025-10-03"
INTERVAL = "15m"
SOURCE = "mt5"
REQUIRED_INSTRUMENTS = {
    "US500": "US SPX 500 Index spot CFD",
    "USTEC": "US Tech 100 Index spot CFD",
}
UNAVAILABLE_FAMILIES = {
    "VIX": "Only a current-expiry VIX futures CFD was available; no stable spot/continuous identity.",
    "DXY": "Only a current-expiry DXY futures CFD was available; no stable spot/continuous identity.",
    "US_TREASURY_YIELDS": "No timestamped cash-yield history was available from the verified MT5 source.",
}


def cache_path(symbol: str, context_dir: Path = CONTEXT_DIR) -> Path:
    return context_dir / f"{symbol}_{REQUESTED_START}_{REQUESTED_END}_{INTERVAL}_{SOURCE}.parquet"


def validate_context_frame(frame: pd.DataFrame, symbol: str) -> dict[str, Any]:
    required = ["Open", "High", "Low", "Close", "Volume"]
    if list(frame.columns) != required:
        raise RuntimeError(f"{symbol} columns differ from required OHLCV schema")
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise RuntimeError(f"{symbol} timestamps must be unique and chronological")
    first = pd.Timestamp(frame.index[0])
    last = pd.Timestamp(frame.index[-1])
    if first > pd.Timestamp("2022-05-19"):
        raise RuntimeError(f"{symbol} begins too late for the research TRAIN window: {first}")
    if last < pd.Timestamp("2025-10-02 12:00:00"):
        raise RuntimeError(f"{symbol} does not cover the validation endpoint: {last}")
    return {
        "row_count": len(frame),
        "first_bar_open_utc": first.isoformat(),
        "last_bar_open_utc": last.isoformat(),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "missing_ohlcv_values": int(frame[required].isna().sum().sum()),
    }


def _acquire_context_data(context_dir: Path) -> dict[str, Any]:
    context_dir.mkdir(parents=True, exist_ok=True)
    ingestion = DataIngestion(data_dir=str(context_dir), strict_research_mode=True)
    instruments = []
    for symbol, identity in REQUIRED_INSTRUMENTS.items():
        frame = ingestion.fetch(
            symbol,
            REQUESTED_START,
            REQUESTED_END,
            interval=INTERVAL,
            source=SOURCE,
            use_cache=True,
            allow_test_fallback=False,
        )
        path = cache_path(symbol, context_dir)
        provenance_path = Path(f"{path}.provenance.json")
        if not path.exists() or not provenance_path.exists():
            raise RuntimeError(f"Context acquisition did not persist data and provenance for {symbol}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("source") != "mt5" or provenance.get("fallback_used") is not False:
            raise RuntimeError(f"Untrusted provenance for {symbol}")
        validation = validate_context_frame(frame, symbol)
        instruments.append(
            {
                "source": "MetaTrader5 via IC Markets terminal",
                "instrument": symbol,
                "instrument_identity": identity,
                "timeframe": "M15",
                "timezone": "UTC",
                "timestamp_resolution": "1 second epoch converted to UTC-naive DatetimeIndex",
                "timestamp_semantics": "bar open",
                "availability_rule": "bar_open_timestamp + 15 minutes",
                "retrieval_date_utc": provenance["retrieved_at_utc"],
                "historical_coverage": {
                    "requested_start": REQUESTED_START,
                    "requested_end": REQUESTED_END,
                    **validation,
                },
                "missing_data_behavior": "No synthetic fill; backward as-of join is capped at 4 hours and records source age.",
                "historical_availability": "Final OHLCV is usable only at the recorded bar close availability timestamp.",
                "revision_policy": "Broker bar snapshot retrieved once; no revised macroeconomic series used.",
                "broker": provenance.get("broker"),
                "server": provenance.get("server"),
                "terminal": provenance.get("terminal"),
                "data_path": str(path),
                "provenance_path": str(provenance_path),
                "data_sha256": sha256(path),
                "provenance_sha256": sha256(provenance_path),
            }
        )
    manifest = {
        "version": CONTEXT_DATA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "instruments": instruments,
        "unavailable_families": {
            name: {"status": "UNAVAILABLE_RELIABLE_HISTORY", "reason": reason}
            for name, reason in UNAVAILABLE_FAMILIES.items()
        },
    }
    manifest_path = context_dir / "context_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_comparable = {key: value for key, value in existing.items() if key != "generated_at_utc"}
        new_comparable = {key: value for key, value in manifest.items() if key != "generated_at_utc"}
        if existing_comparable != new_comparable:
            raise FileExistsError("Existing context manifest differs; refusing overwrite")
        return existing
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def acquire_context_data(context_dir: Path = CONTEXT_DIR) -> dict[str, Any]:
    context_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = context_dir.parent / f".{context_dir.name}.acquisition.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Context acquisition is already active: {lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
        return _acquire_context_data(context_dir)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print(json.dumps(acquire_context_data(), indent=2))