"""Acquire the approved free external information through VALIDATION only."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from diagnose_baseline import DATASET_PATH, METADATA_PATH, sha256
from src.external_information import (
    SCHEMA_COLUMNS,
    atomic_write_json,
    atomic_write_parquet,
    estimate_frame_mib,
    normalize_observations,
    sha256_bytes,
    sha256_file,
)


DATASET_VERSION = "free_external_v1"
OUTPUT_ROOT = Path("data/information_sources") / DATASET_VERSION
BASELINE_DIR = Path("models/baselines")
EXPERIMENT_OUTPUT = BASELINE_DIR / "v0.3_forward_atr_xgb_baseline_v1_free_external_information_v1"
START_UTC = pd.Timestamp("2022-05-12T00:00:00Z")
NEW_YORK = ZoneInfo("America/New_York")
MARKET_SYMBOLS = {
    "EURUSD": "Euro vs US Dollar",
    "GBPUSD": "Great Britain Pound vs US Dollar",
    "USDJPY": "US Dollar vs Japanese Yen",
    "USDCHF": "US Dollar vs Swiss Franc",
    "XAUUSD": "Gold vs US Dollar",
    "XAGUSD": "Silver vs US Dollar",
    "XTIUSD": "Crude Oil vs US Dollar",
    "XBRUSD": "Brent Oil vs US Dollar",
    "US2000": "US Small Cap 2000 Index",
}
TREASURY_FIELDS = {
    "BC_2YEAR": "2Y",
    "BC_5YEAR": "5Y",
    "BC_10YEAR": "10Y",
    "BC_30YEAR": "30Y",
}
CBOE_SYMBOLS = ("VIX", "VIX9D", "VIX3M", "VVIX")


def validation_end_utc() -> pd.Timestamp:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pd.Timestamp(metadata["splits"]["validation"]["last_timestamp"], tz="UTC") + pd.Timedelta(minutes=15)


def protected_hashes() -> dict[str, str]:
    paths = [DATASET_PATH, METADATA_PATH]
    paths.extend(
        path
        for root in (Path("data/raw"), Path("data/context"), BASELINE_DIR)
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and EXPERIMENT_OUTPUT not in path.parents
    )
    return {
        str(path).replace("\\", "/"): sha256(path)
        for path in sorted(set(paths), key=lambda item: str(item))
    }


def resource_preflight() -> dict[str, Any]:
    expected_market_rows = len(MARKET_SYMBOLS) * 100_000
    market_mib = estimate_frame_mib(expected_market_rows, 2, 2)
    return {
        "estimated_rows": {
            "mt5_maximum": expected_market_rows,
            "treasury": 3_600,
            "cboe": 4_500,
            "calendar": 1_300,
            "macro": 0,
        },
        "estimated_disk_mib": round(market_mib * 0.35 + 5, 2),
        "estimated_peak_ram_mib": 220,
        "market_chunk_days": 31,
        "decision": "SAFE" if market_mib < 1024 else "STOP_UNSAFE_MEMORY",
    }


def _write_bytes(content: bytes, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable raw data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _observation_rows(
    *,
    source: str,
    instrument: str,
    field: str,
    values: Iterable[float],
    source_times: Iterable[pd.Timestamp],
    available_times: Iterable[pd.Timestamp],
    received_time: datetime,
    raw_hash: str,
    quality_flags: str,
) -> pd.DataFrame:
    values_list = list(values)
    source_list = list(source_times)
    available_list = list(available_times)
    return pd.DataFrame(
        {
            "source": source,
            "dataset_version": DATASET_VERSION,
            "instrument_or_event_id": instrument,
            "field": field,
            "value": values_list,
            "source_time_utc": source_list,
            "available_time_utc": available_list,
            "received_time_utc": received_time,
            "revision_id": "initial",
            "quality_flags": quality_flags,
            "raw_batch_sha256": raw_hash,
        },
        columns=SCHEMA_COLUMNS,
    )


def acquire_treasury(root: Path, session: requests.Session) -> dict[str, Any]:
    raw_dir = root / "treasury" / "raw"
    frames: list[pd.DataFrame] = []
    raw_files = []
    received = datetime.now(timezone.utc)
    cutoff = validation_end_utc()
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }
    for year in range(START_UTC.year, cutoff.year + 1):
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
        )
        response = session.get(url, timeout=30)
        response.raise_for_status()
        content = response.content
        raw_path = raw_dir / f"daily_treasury_yield_curve_{year}.xml"
        _write_bytes(content, raw_path)
        raw_hash = sha256_bytes(content)
        raw_files.append({"path": str(raw_path), "sha256": raw_hash, "bytes": len(content), "url": url})
        document = ET.fromstring(content)
        for entry in document.findall("atom:entry", namespaces):
            properties = entry.find("atom:content/m:properties", namespaces)
            if properties is None:
                continue
            values = {child.tag.rsplit("}", 1)[-1]: child.text for child in properties}
            if not values.get("NEW_DATE"):
                continue
            source_date = pd.Timestamp(values["NEW_DATE"]).date()
            source_time = pd.Timestamp(datetime.combine(source_date, time(15, 30), NEW_YORK)).tz_convert("UTC")
            available = pd.Timestamp(datetime.combine(source_date, time(18, 0), NEW_YORK)).tz_convert("UTC")
            if available > cutoff:
                continue
            for source_field, maturity in TREASURY_FIELDS.items():
                value = values.get(source_field)
                if value in (None, ""):
                    continue
                frames.append(
                    _observation_rows(
                        source="US_TREASURY",
                        instrument=maturity,
                        field="yield_percent",
                        values=[float(value)],
                        source_times=[source_time],
                        available_times=[available],
                        received_time=received,
                        raw_hash=raw_hash,
                        quality_flags="conservative_18et_availability",
                    )
                )
    normalized = normalize_observations(pd.concat(frames, ignore_index=True))
    output = root / "treasury" / "normalized" / "treasury_yields.parquet"
    atomic_write_parquet(normalized, output)
    return {
        "status": "ACQUIRED",
        "source": "US Treasury official XML",
        "rows": len(normalized),
        "data_path": str(output),
        "data_sha256": sha256_file(output),
        "raw_files": raw_files,
        "availability": "indicative quote at 15:30 ET; conservatively usable at 18:00 ET",
    }


def acquire_cboe(root: Path, session: requests.Session) -> dict[str, Any]:
    frames = []
    instruments = []
    received = datetime.now(timezone.utc)
    cutoff = validation_end_utc()
    for symbol in CBOE_SYMBOLS:
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        content = response.content
        raw_path = root / "cboe" / "raw" / f"{symbol}_History.csv"
        _write_bytes(content, raw_path)
        raw_hash = sha256_bytes(content)
        raw = pd.read_csv(io.BytesIO(content))
        raw.columns = [str(column).strip().upper() for column in raw.columns]
        value_column = "VVIX" if symbol == "VVIX" else "CLOSE"
        dates = pd.to_datetime(raw["DATE"], format="%m/%d/%Y", errors="raise")
        source_times = dates.map(
            lambda item: pd.Timestamp(datetime.combine(item.date(), time(16, 15), NEW_YORK)).tz_convert("UTC")
        )
        available_times = dates.map(
            lambda item: pd.Timestamp(datetime.combine(item.date(), time(17, 0), NEW_YORK)).tz_convert("UTC")
        )
        eligible = available_times <= cutoff
        frame = _observation_rows(
            source="CBOE",
            instrument=symbol,
            field="close",
            values=pd.to_numeric(raw.loc[eligible, value_column], errors="raise"),
            source_times=source_times[eligible],
            available_times=available_times[eligible],
            received_time=received,
            raw_hash=raw_hash,
            quality_flags="daily_close_conservative_17et_availability",
        )
        frames.append(frame)
        instruments.append({"instrument": symbol, "raw_sha256": raw_hash, "raw_bytes": len(content), "url": url})
    normalized = normalize_observations(pd.concat(frames, ignore_index=True))
    output = root / "cboe" / "normalized" / "volatility_indices.parquet"
    atomic_write_parquet(normalized, output)
    return {
        "status": "ACQUIRED",
        "source": "Cboe official daily history",
        "rows": len(normalized),
        "data_path": str(output),
        "data_sha256": sha256_file(output),
        "instruments": instruments,
        "availability": "daily close conservatively usable at 17:00 ET",
    }


def _observed(date_value: date) -> date:
    if date_value.weekday() == 5:
        return date_value - timedelta(days=1)
    if date_value.weekday() == 6:
        return date_value + timedelta(days=1)
    return date_value


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def exchange_holidays(year: int) -> set[date]:
    from dateutil.easter import easter

    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year == 2025:
        holidays.add(date(2025, 1, 9))
    return holidays


EARLY_CLOSES = {
    date(2022, 11, 25),
    date(2023, 7, 3), date(2023, 11, 24),
    date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
}


def acquire_calendar(root: Path) -> dict[str, Any]:
    cutoff = validation_end_utc()
    dates = pd.date_range(START_UTC.normalize(), cutoff.normalize(), freq="D", tz="UTC")
    holiday_dates = set().union(*(exchange_holidays(year) for year in range(START_UTC.year, cutoff.year + 1)))
    sessions = pd.DataFrame(index=dates)
    sessions["is_trading_day"] = [timestamp.weekday() < 5 and timestamp.date() not in holiday_dates for timestamp in dates]
    sessions["holiday"] = [timestamp.date() in holiday_dates for timestamp in dates]
    sessions["early_close"] = [timestamp.date() in EARLY_CLOSES for timestamp in dates]
    trading_dates = sessions.index[sessions["is_trading_day"]]
    sessions["day_before_holiday"] = [
        bool(is_trading and any((timestamp + pd.Timedelta(days=days)).date() in holiday_dates for days in (1, 2, 3)))
        for timestamp, is_trading in sessions["is_trading_day"].items()
    ]
    trading = pd.Series(trading_dates, index=trading_dates)
    month_first = set(trading.groupby([trading.dt.year, trading.dt.month]).first())
    month_last = set(trading.groupby([trading.dt.year, trading.dt.month]).last())
    quarter_last = set(trading.groupby([trading.dt.year, trading.dt.quarter]).last())
    year_last = set(trading.groupby(trading.dt.year).last())
    sessions["first_trading_day"] = sessions.index.isin(month_first)
    sessions["last_trading_day"] = sessions.index.isin(month_last)
    sessions["month_end"] = sessions.index.isin(month_last)
    sessions["quarter_end"] = sessions.index.isin(quarter_last)
    sessions["year_end"] = sessions.index.isin(year_last)
    raw_payload = {
        "source": "NYSE official hours/calendar rules",
        "source_url": "https://www.nyse.com/trade/hours-calendars",
        "dataset_version": DATASET_VERSION,
        "holidays": sorted(item.isoformat() for item in holiday_dates),
        "early_closes": sorted(item.isoformat() for item in EARLY_CLOSES if START_UTC.date() <= item <= cutoff.date()),
        "special_closures": ["2025-01-09"],
    }
    raw_content = json.dumps(raw_payload, sort_keys=True).encode("utf-8")
    raw_path = root / "calendar" / "raw" / "nyse_calendar_rules.json"
    _write_bytes(raw_content, raw_path)
    raw_hash = sha256_bytes(raw_content)
    frames = []
    received = datetime.now(timezone.utc)
    for field in sessions.columns:
        frames.append(
            _observation_rows(
                source="NYSE",
                instrument="XNYS_CALENDAR",
                field=field,
                values=sessions[field].astype(float),
                source_times=sessions.index,
                available_times=sessions.index,
                received_time=received,
                raw_hash=raw_hash,
                quality_flags="deterministic_calendar_available_at_day_start",
            )
        )
    normalized = normalize_observations(pd.concat(frames, ignore_index=True))
    output = root / "calendar" / "normalized" / "nyse_calendar.parquet"
    atomic_write_parquet(normalized, output)
    return {
        "status": "ACQUIRED",
        "source": "NYSE official calendar rules",
        "rows": len(normalized),
        "data_path": str(output),
        "data_sha256": sha256_file(output),
        "raw_sha256": raw_hash,
    }


def _mt5_frame(rates: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(rates)
    frame["bar_open_utc"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    return frame


def acquire_mt5(root: Path) -> dict[str, Any]:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        return {"status": "UNAVAILABLE_RELIABLE_HISTORY", "reason": f"MT5 initialize failed: {mt5.last_error()}"}
    cutoff = validation_end_utc()
    accepted = []
    rejected = []
    try:
        for symbol, expected_description in MARKET_SYMBOLS.items():
            info = mt5.symbol_info(symbol)
            if info is None or not mt5.symbol_select(symbol, True):
                rejected.append({"instrument": symbol, "reason": "symbol unavailable or not selectable"})
                continue
            if info.description != expected_description or info.expiration_time != 0:
                rejected.append({"instrument": symbol, "reason": "identity or expiry gate failed", "description": info.description, "expiration_time": info.expiration_time})
                continue
            windows = (
                (START_UTC, START_UTC + pd.Timedelta(days=7)),
                (pd.Timestamp("2024-01-08T00:00:00Z"), pd.Timestamp("2024-01-15T00:00:00Z")),
                (cutoff - pd.Timedelta(days=7), cutoff),
            )
            probes = [mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start.to_pydatetime(), end.to_pydatetime()) for start, end in windows]
            if any(item is None or len(item) == 0 for item in probes):
                rejected.append({"instrument": symbol, "reason": "coverage probe returned no rows", "counts": [0 if item is None else len(item) for item in probes]})
                continue
            if pd.to_datetime(probes[-1][-1]["time"], unit="s", utc=True) < cutoff - pd.Timedelta(minutes=15):
                rejected.append({"instrument": symbol, "reason": "validation endpoint not covered"})
                continue
            chunk_start = START_UTC
            chunks = []
            normalized_frames = []
            while chunk_start < cutoff:
                chunk_end = min(chunk_start + pd.Timedelta(days=31), cutoff)
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, chunk_start.to_pydatetime(), chunk_end.to_pydatetime())
                if rates is not None and len(rates):
                    frame = _mt5_frame(rates)
                    frame = frame[(frame["bar_open_utc"] >= chunk_start) & (frame["bar_open_utc"] + pd.Timedelta(minutes=15) <= cutoff)]
                    frame = frame.drop_duplicates("bar_open_utc", keep="last").sort_values("bar_open_utc")
                    raw_path = root / "mt5" / "raw" / symbol / f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}_M15.parquet"
                    atomic_write_parquet(frame, raw_path)
                    raw_hash = sha256_file(raw_path)
                    chunks.append({"path": str(raw_path), "sha256": raw_hash, "rows": len(frame)})
                    normalized_frames.append(
                        _observation_rows(
                            source="IC_MARKETS_MT5",
                            instrument=symbol,
                            field="close",
                            values=frame["close"],
                            source_times=frame["bar_open_utc"],
                            available_times=frame["bar_open_utc"] + pd.Timedelta(minutes=15),
                            received_time=datetime.now(timezone.utc),
                            raw_hash=raw_hash,
                            quality_flags="broker_m15_closed_bar",
                        )
                    )
                chunk_start = chunk_end
            normalized = normalize_observations(pd.concat(normalized_frames, ignore_index=True))
            if normalized["available_time_utc"].max() > cutoff:
                raise RuntimeError(f"{symbol} acquisition crossed the validation boundary")
            output = root / "mt5" / "normalized" / f"{symbol}_M15_close.parquet"
            atomic_write_parquet(normalized, output)
            accepted.append({
                "instrument": symbol,
                "description": info.description,
                "path": info.path,
                "expiration_time": info.expiration_time,
                "rows": len(normalized),
                "first_available_time_utc": normalized["available_time_utc"].min().isoformat(),
                "last_available_time_utc": normalized["available_time_utc"].max().isoformat(),
                "missing_close": int(normalized["value"].isna().sum()),
                "data_path": str(output),
                "data_sha256": sha256_file(output),
                "chunks": chunks,
            })
    finally:
        mt5.shutdown()
    return {
        "status": "ACQUIRED" if accepted else "UNAVAILABLE_RELIABLE_HISTORY",
        "source": "IC Markets MetaTrader5",
        "accepted": accepted,
        "rejected": rejected,
    }


def macro_status() -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE_RELIABLE_HISTORY",
        "reason": (
            "No FRED_API_KEY is configured and the BLS schedule endpoint is blocked. "
            "Current official values cannot substitute for historical initial vintages; "
            "historical consensus is also unavailable for free."
        ),
        "consensus": "UNAVAILABLE_RELIABLE_HISTORY",
        "rows": 0,
    }


def _rebase_paths(value: Any, staging: Path, output_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _rebase_paths(item, staging, output_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_paths(item, staging, output_root) for item in value]
    if isinstance(value, str):
        return value.replace(str(staging), str(output_root))
    return value


def run_acquisition(*, include_mt5: bool = True, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset version: {output_root}")
    resources = resource_preflight()
    if resources["decision"] != "SAFE":
        raise MemoryError("Projected acquisition memory is unsafe")
    before = protected_hashes()
    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    session = requests.Session()
    session.headers["User-Agent"] = "ML-Trading-System/1.0 private research"
    started = datetime.now(timezone.utc)
    try:
        sources = {
            "macro": macro_status(),
            "treasury": acquire_treasury(staging, session),
            "volatility": acquire_cboe(staging, session),
            "calendar": acquire_calendar(staging),
            "mt5": acquire_mt5(staging) if include_mt5 else {"status": "SKIPPED", "reason": "MT5 disabled by command option"},
        }
        sources = _rebase_paths(sources, staging, output_root)
        manifest = {
            "dataset_version": DATASET_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started.isoformat(),
            "scope": "TRAIN and VALIDATION only; TEST rows and labels not accessed",
            "first_requested_time_utc": START_UTC.isoformat(),
            "last_allowed_available_time_utc": validation_end_utc().isoformat(),
            "resource_preflight": resources,
            "normalized_schema": SCHEMA_COLUMNS,
            "sources": sources,
            "protected_hashes_before": before,
        }
        after = protected_hashes()
        if before != after:
            raise RuntimeError("A protected artifact changed during acquisition")
        manifest["protected_hashes_after"] = after
        manifest["protected_artifacts_unchanged"] = True
        atomic_write_json(manifest, staging / "manifest.json")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mt5", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run_acquisition(include_mt5=not arguments.skip_mt5), indent=2))