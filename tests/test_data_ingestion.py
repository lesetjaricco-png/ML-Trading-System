from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from main import resolve_use_cache
from src.data_ingestion import DataIngestion, IngestionError


def _make_mt5_rates() -> list[dict]:
    return [
        {
            "time": 1704067200,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 1200,
        },
        {
            "time": 1704068100,
            "open": 100.5,
            "high": 101.2,
            "low": 100.1,
            "close": 100.9,
            "tick_volume": 1100,
        },
    ]


def _install_fake_mt5(
    monkeypatch,
    *,
    initialize_ok: bool = True,
    symbol_available: bool = True,
    symbol_select_ok: bool = True,
    rates: list[dict] | None = None,
):
    fake_mt5 = types.ModuleType("MetaTrader5")
    fake_mt5.TIMEFRAME_M15 = 15
    fake_mt5.initialize = lambda *args, **kwargs: initialize_ok
    fake_mt5.shutdown = lambda: None
    fake_mt5.symbol_info = lambda *args, **kwargs: object() if symbol_available else None
    fake_mt5.symbol_select = lambda *args, **kwargs: symbol_select_ok
    fake_mt5.copy_rates_range = lambda *args, **kwargs: rates if rates is not None else []
    fake_mt5.copy_rates_from = lambda *args, **kwargs: rates if rates is not None else []
    fake_mt5.last_error = lambda: (-2, "Terminal: Invalid params")
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)


def _write_cache_with_metadata(
    ingestion: DataIngestion,
    df: pd.DataFrame,
    cache_path,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    requested_start: str,
    requested_end: str,
    fallback_used: bool,
):
    df.to_parquet(cache_path)
    metadata = {
        "schema_version": ingestion.PROVENANCE_SCHEMA_VERSION,
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": pd.Timestamp(df.index.min()).isoformat(),
        "actual_end": pd.Timestamp(df.index.max()).isoformat(),
        "actual_first_timestamp": pd.Timestamp(df.index.min()).isoformat(),
        "actual_last_timestamp": pd.Timestamp(df.index.max()).isoformat(),
        "row_count": int(len(df)),
        "chunk_count": 1,
        "duplicate_count": 0,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "from_cache": False,
        "cache_source_provenance": None,
        "fallback_used": fallback_used,
    }
    metadata_path = ingestion._cache_metadata_path(str(cache_path))
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


def test_fetch_mt5_success_returns_mt5_data_with_provenance(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, rates=_make_mt5_rates())

    ingestion = DataIngestion(data_dir=str(tmp_path))
    df = ingestion.fetch(
        ticker="US30",
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="15m",
        source="mt5",
        use_cache=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.iloc[0]["Close"] == 100.5
    assert df.attrs["provenance"]["source"] == "mt5"
    assert df.attrs["provenance"]["fallback_used"] is False
    assert df.attrs["provenance"]["chunk_count"] == 1
    assert df.attrs["provenance"]["duplicate_count"] == 0
    assert df.attrs["provenance"]["actual_first_timestamp"]
    assert df.attrs["provenance"]["actual_last_timestamp"]


def test_fetch_mt5_failure_raises_explicit_error(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, initialize_ok=False)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    with pytest.raises(IngestionError, match="initialization failed"):
        ingestion.fetch(
            ticker="US30",
            start_date="2024-01-01",
            end_date="2024-01-02",
            interval="15m",
            source="mt5",
            use_cache=False,
        )


def test_fallback_requires_explicit_test_only_flag(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, initialize_ok=False)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    with pytest.raises(IngestionError):
        ingestion.fetch(
            ticker="US30",
            start_date="2024-01-01",
            end_date="2024-01-02",
            interval="15m",
            source="mt5",
            use_cache=False,
        )

    df = ingestion.fetch(
        ticker="US30",
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="15m",
        source="mt5",
        use_cache=False,
        allow_test_fallback=True,
    )
    assert df.attrs["provenance"]["source"] == "fallback"
    assert df.attrs["provenance"]["fallback_used"] is True


def test_fallback_generated_cache_is_rejected(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, initialize_ok=False)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    cache_path = tmp_path / "US30_2024-01-01_2024-01-02_15m_mt5.parquet"
    fallback_df = ingestion._build_fallback_dataframe("US30", "2024-01-01", "2024-01-02", "15m")
    _write_cache_with_metadata(
        ingestion,
        fallback_df,
        cache_path,
        source="fallback",
        symbol="US30",
        timeframe="15m",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        fallback_used=True,
    )

    with pytest.raises(IngestionError):
        ingestion.fetch(
            ticker="US30",
            start_date="2024-01-01",
            end_date="2024-01-02",
            interval="15m",
            source="mt5",
            use_cache=True,
        )


def test_cache_without_provenance_metadata_is_rejected(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, initialize_ok=False)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    cache_path = tmp_path / "US30_2024-01-01_2024-01-02_15m_mt5.parquet"
    fallback_df = ingestion._build_fallback_dataframe("US30", "2024-01-01", "2024-01-02", "15m")
    fallback_df.to_parquet(cache_path)

    with pytest.raises(IngestionError):
        ingestion.fetch(
            ticker="US30",
            start_date="2024-01-01",
            end_date="2024-01-02",
            interval="15m",
            source="mt5",
            use_cache=True,
        )


def test_valid_mt5_cache_can_be_reused(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, rates=_make_mt5_rates())

    ingestion = DataIngestion(data_dir=str(tmp_path))
    first = ingestion.fetch(
        ticker="US30",
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="15m",
        source="mt5",
        use_cache=False,
    )
    assert first.attrs["provenance"]["from_cache"] is False

    _install_fake_mt5(monkeypatch, initialize_ok=False)
    second = ingestion.fetch(
        ticker="US30",
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="15m",
        source="mt5",
        use_cache=True,
    )
    assert second.attrs["provenance"]["from_cache"] is True
    assert len(second) == len(first)


def test_invalid_symbol_or_timeframe_provenance_is_rejected(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, initialize_ok=False)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    cache_path = tmp_path / "US30_2024-01-01_2024-01-02_15m_mt5.parquet"
    df = ingestion._build_fallback_dataframe("US30", "2024-01-01", "2024-01-02", "15m")
    _write_cache_with_metadata(
        ingestion,
        df,
        cache_path,
        source="mt5",
        symbol="US500",
        timeframe="1h",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        fallback_used=False,
    )

    with pytest.raises(IngestionError):
        ingestion.fetch(
            ticker="US30",
            start_date="2024-01-01",
            end_date="2024-01-02",
            interval="15m",
            source="mt5",
            use_cache=True,
        )


def test_invalid_ohlc_data_is_rejected(tmp_path, monkeypatch):
    _install_fake_mt5(monkeypatch, initialize_ok=False)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    cache_path = tmp_path / "US30_2024-01-01_2024-01-02_15m_mt5.parquet"
    index = pd.date_range("2024-01-01", periods=2, freq="15min")
    bad_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [99.0, 100.0],
            "Low": [99.5, 100.5],
            "Close": [100.5, 100.8],
            "Volume": [1000, 1200],
        },
        index=index,
    )
    bad_df.index.name = "Date"
    _write_cache_with_metadata(
        ingestion,
        bad_df,
        cache_path,
        source="mt5",
        symbol="US30",
        timeframe="15m",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        fallback_used=False,
    )

    with pytest.raises(IngestionError):
        ingestion.fetch(
            ticker="US30",
            start_date="2024-01-01",
            end_date="2024-01-02",
            interval="15m",
            source="mt5",
            use_cache=True,
        )


def test_resolve_use_cache_prefers_config_and_cli_override():
    cfg = {"data": {"use_cache": False}}

    assert resolve_use_cache(cfg, True) is True
    assert resolve_use_cache(cfg, False) is False

    cfg = {"data": {}}
    assert resolve_use_cache(cfg, True) is True
    assert resolve_use_cache(cfg, False) is False


class _ChunkingMt5:
    def __init__(self, *, response: str = "rates", coverage_delay_days: int = 0):
        self.response = response
        self.coverage_delay_days = coverage_delay_days
        self.calls: list[tuple[str, int, datetime, datetime]] = []

    def copy_rates_range(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe, start, end))
        if self.response == "none":
            return None
        if self.response == "empty":
            return []

        actual_start = start + timedelta(days=self.coverage_delay_days)
        timestamps = [actual_start]
        if len(self.calls) > 1:
            timestamps.append(start + timedelta(minutes=15))
        timestamps.append(end)
        return [
            {
                "time": int(timestamp.timestamp()),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "tick_volume": 1000,
            }
            for timestamp in timestamps
        ]

    def last_error(self):
        return (-2, "Terminal: Invalid params")


@pytest.mark.parametrize("bar_count", [10, 100])
def test_mt5_small_requests_use_one_bounded_chunk(bar_count):
    ingestion = DataIngestion(data_dir="data/raw")
    mt5 = _ChunkingMt5()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=15 * (bar_count - 1))

    rates, metadata = ingestion._fetch_mt5_rates(mt5, "US30", 15, "15m", start, end)

    assert len(mt5.calls) == 1
    assert metadata == {"chunk_count": 1, "duplicate_count": 0}
    assert rates["time"].is_monotonic_increasing


def test_mt5_large_range_is_chunked_below_safe_limit_and_deduplicated():
    ingestion = DataIngestion(data_dir="data/raw")
    mt5 = _ChunkingMt5()
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 9, tzinfo=timezone.utc)

    rates, metadata = ingestion._fetch_mt5_rates(mt5, "US30", 15, "15m", start, end)

    assert len(mt5.calls) > 1
    theoretical_chunk_sizes = []
    for symbol, timeframe, chunk_start, chunk_end in mt5.calls:
        theoretical_bars = int((chunk_end - chunk_start) / timedelta(minutes=15)) + 1
        theoretical_chunk_sizes.append(theoretical_bars)
        assert symbol == "US30"
        assert timeframe == 15
        assert theoretical_bars <= ingestion.MT5_SAFE_CHUNK_BARS
        assert chunk_start.tzinfo is timezone.utc
        assert chunk_end.tzinfo is timezone.utc
    assert max(theoretical_chunk_sizes) == ingestion.MT5_SAFE_CHUNK_BARS
    assert metadata["chunk_count"] == len(mt5.calls)
    assert metadata["duplicate_count"] == len(mt5.calls) - 1
    assert rates["time"].is_monotonic_increasing
    assert rates["time"].is_unique


@pytest.mark.parametrize("response", ["none", "empty"])
def test_mt5_none_or_empty_chunk_raises_explicit_error(response):
    ingestion = DataIngestion(data_dir="data/raw")
    mt5 = _ChunkingMt5(response=response)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(IngestionError, match="returned no rates"):
        ingestion._fetch_mt5_rates(
            mt5, "US30", 15, "15m", start, start + timedelta(days=1)
        )


def test_mt5_insufficient_start_coverage_raises_explicit_error():
    ingestion = DataIngestion(data_dir="data/raw")
    mt5 = _ChunkingMt5(coverage_delay_days=30)
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(IngestionError, match="Insufficient MT5 history"):
        ingestion._fetch_mt5_rates(
            mt5, "US30", 15, "15m", start, start + timedelta(days=60)
        )
