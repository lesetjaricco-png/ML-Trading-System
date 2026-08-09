"""Data ingestion module: fetch and cache OHLCV market data."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when ingestion cannot retrieve validated market data."""


class DataProvenanceError(IngestionError):
    """Raised when cache/data provenance is invalid for research usage."""


class DataIngestion:
    """Downloads and caches historical OHLCV data from MT5/IC Markets or yfinance."""

    PROVENANCE_SCHEMA_VERSION = 1
    MT5_SAFE_CHUNK_BARS = 90_000
    MT5_COVERAGE_TOLERANCE = timedelta(days=7)

    def __init__(self, data_dir: str = "data/raw", strict_research_mode: bool = True):
        self.data_dir = data_dir
        self.strict_research_mode = strict_research_mode
        os.makedirs(data_dir, exist_ok=True)

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "15m",
        source: str = "mt5",
        use_cache: bool = True,
        allow_test_fallback: bool = False,
    ) -> pd.DataFrame:
        """Return OHLCV data for *ticker* between *start_date* and *end_date*.

        Parameters
        ----------
        ticker:
            Stock ticker symbol, e.g. ``"AAPL"``.
        start_date:
            ISO-8601 date string, e.g. ``"2018-01-01"``.
        end_date:
            ISO-8601 date string, e.g. ``"2024-01-01"``.
        interval:
            Data frequency supported by yfinance (``"1d"``, ``"1h"`` …).
        use_cache:
            When *True* (default) a local Parquet file is used if available.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``Open, High, Low, Close, Volume`` and a
            ``DatetimeIndex``.
        """
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)

        cache_path = os.path.join(
            self.data_dir, f"{ticker}_{start_date}_{end_date}_{interval}_{source}.parquet"
        )
        metadata_path = self._cache_metadata_path(cache_path)

        if use_cache and os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            cached = self._try_load_valid_cache(
                cache_path=cache_path,
                metadata_path=metadata_path,
                expected_symbol=ticker,
                expected_timeframe=interval,
                expected_source=source,
                requested_start=start_date,
                requested_end=end_date,
            )
            if cached is not None:
                return cached

        source_value = source.lower()
        fallback_used = False
        if source_value == "mt5":
            try:
                df = self._fetch_from_mt5(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )
            except IngestionError:
                if not allow_test_fallback:
                    raise
                logger.warning(
                    "MT5 ingestion failed; explicit test fallback mode enabled. Returning fallback sample data."
                )
                df = self._build_fallback_dataframe(ticker, start_date, end_date, interval)
                fallback_used = True
        else:
            logger.info("Downloading %s data from yfinance …", ticker)
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                raise ValueError(
                    f"No data returned for ticker '{ticker}' "
                    f"({start_date} – {end_date}, interval={interval})."
                )

            # Flatten multi-level column index if present (multi-ticker download)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)

        df = self._finalize_dataframe(
            df,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        provenance = self._build_provenance(
            source="fallback" if fallback_used else source_value,
            symbol=ticker,
            timeframe=interval,
            requested_start=start_date,
            requested_end=end_date,
            df=df,
            from_cache=False,
            fallback_used=fallback_used,
            cache_source_provenance=None,
        )
        self._validate_research_data(
            df,
            provenance,
            expected_symbol=ticker,
            expected_timeframe=interval,
            expected_source=source_value,
            requested_start=start_date,
            requested_end=end_date,
            allow_fallback=allow_test_fallback,
        )
        self._write_cache_with_provenance(df, cache_path, metadata_path, provenance)
        df.attrs["provenance"] = provenance
        return df

    def _finalize_dataframe(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        df = df.copy()
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError(
                f"No usable OHLCV rows available for ticker '{ticker}' ({start_date} – {end_date}, interval={interval})."
            )

        return df

    def _cache_metadata_path(self, cache_path: str) -> str:
        return f"{cache_path}.provenance.json"

    def _try_load_valid_cache(
        self,
        *,
        cache_path: str,
        metadata_path: str,
        expected_symbol: str,
        expected_timeframe: str,
        expected_source: str,
        requested_start: str,
        requested_end: str,
    ) -> pd.DataFrame | None:
        try:
            cached_df = pd.read_parquet(cache_path)
        except Exception as exc:
            logger.warning("Failed to read cached parquet at %s: %s", cache_path, exc)
            return None

        try:
            cache_metadata = self._read_cache_metadata(metadata_path)
            self._validate_cache_metadata(
                cache_metadata,
                expected_symbol=expected_symbol,
                expected_timeframe=expected_timeframe,
                expected_source=expected_source,
                requested_start=requested_start,
                requested_end=requested_end,
            )
            provenance = dict(cache_metadata)
            provenance["from_cache"] = True
            provenance["cache_source_provenance"] = dict(cache_metadata)
            self._validate_research_data(
                cached_df,
                provenance,
                expected_symbol=expected_symbol,
                expected_timeframe=expected_timeframe,
                expected_source=expected_source,
                requested_start=requested_start,
                requested_end=requested_end,
            )
        except IngestionError as exc:
            logger.warning("Rejecting cached dataset at %s: %s", cache_path, exc)
            return None

        cached_df.attrs["provenance"] = provenance
        logger.info("Using validated cache %s with %d rows.", cache_path, len(cached_df))
        return cached_df

    def _read_cache_metadata(self, metadata_path: str) -> dict[str, Any]:
        if not os.path.exists(metadata_path):
            raise DataProvenanceError(
                f"Missing provenance metadata file for cache: {metadata_path}."
            )
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise DataProvenanceError(
                f"Unable to read provenance metadata at {metadata_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise DataProvenanceError(f"Invalid provenance metadata payload in {metadata_path}.")
        return payload

    def _validate_cache_metadata(
        self,
        metadata: dict[str, Any],
        *,
        expected_symbol: str,
        expected_timeframe: str,
        expected_source: str,
        requested_start: str,
        requested_end: str,
        allow_fallback: bool = False,
    ) -> None:
        required_keys = {
            "schema_version",
            "source",
            "symbol",
            "timeframe",
            "requested_start",
            "requested_end",
            "actual_start",
            "actual_end",
            "actual_first_timestamp",
            "actual_last_timestamp",
            "row_count",
            "chunk_count",
            "duplicate_count",
            "retrieved_at_utc",
            "fallback_used",
        }
        missing = sorted(required_keys - set(metadata.keys()))
        if missing:
            raise DataProvenanceError(
                f"Cache provenance is missing required fields: {', '.join(missing)}"
            )

        if metadata.get("schema_version") != self.PROVENANCE_SCHEMA_VERSION:
            raise DataProvenanceError(
                "Cache provenance schema version is unsupported."
            )
        if str(metadata.get("source", "")).lower() != expected_source.lower():
            raise DataProvenanceError("Cache provenance source does not match requested source.")
        if str(metadata.get("symbol", "")) != expected_symbol:
            raise DataProvenanceError("Cache provenance symbol does not match requested symbol.")
        if str(metadata.get("timeframe", "")) != expected_timeframe:
            raise DataProvenanceError("Cache provenance timeframe does not match requested timeframe.")
        if str(metadata.get("requested_start", "")) != requested_start:
            raise DataProvenanceError("Cache provenance requested_start does not match request.")
        if str(metadata.get("requested_end", "")) != requested_end:
            raise DataProvenanceError("Cache provenance requested_end does not match request.")
        if bool(metadata.get("fallback_used")):
            raise DataProvenanceError("Cache provenance indicates fallback data usage.")

    def _build_provenance(
        self,
        *,
        source: str,
        symbol: str,
        timeframe: str,
        requested_start: str,
        requested_end: str,
        df: pd.DataFrame,
        from_cache: bool,
        fallback_used: bool,
        cache_source_provenance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        mt5_retrieval = df.attrs.get("mt5_retrieval", {})
        return {
            "schema_version": self.PROVENANCE_SCHEMA_VERSION,
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "mt5_timeframe": mt5_retrieval.get("mt5_timeframe"),
            "requested_start": requested_start,
            "requested_end": requested_end,
            "actual_start": pd.Timestamp(df.index.min()).isoformat(),
            "actual_end": pd.Timestamp(df.index.max()).isoformat(),
            "actual_first_timestamp": pd.Timestamp(df.index.min()).isoformat(),
            "actual_last_timestamp": pd.Timestamp(df.index.max()).isoformat(),
            "row_count": int(len(df)),
            "chunk_count": int(mt5_retrieval.get("chunk_count", 1)),
            "duplicate_count": int(mt5_retrieval.get("duplicate_count", 0)),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "from_cache": from_cache,
            "cache_source_provenance": cache_source_provenance,
            "fallback_used": bool(fallback_used),
            "broker": mt5_retrieval.get("broker"),
            "server": mt5_retrieval.get("server"),
            "terminal": mt5_retrieval.get("terminal"),
        }

    def _write_cache_with_provenance(
        self,
        df: pd.DataFrame,
        cache_path: str,
        metadata_path: str,
        provenance: dict[str, Any],
    ) -> None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path)
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(provenance, handle, indent=2)
        except Exception as exc:
            logger.warning("Unable to write cache/provenance files for %s: %s", cache_path, exc)
        else:
            logger.info("Saved %d rows to %s", len(df), cache_path)

    def _validate_research_data(
        self,
        df: pd.DataFrame,
        provenance: dict[str, Any],
        *,
        expected_symbol: str,
        expected_timeframe: str,
        expected_source: str,
        requested_start: str,
        requested_end: str,
        allow_fallback: bool = False,
    ) -> None:
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise IngestionError(
                f"Dataset is missing required OHLCV columns: {', '.join(missing_columns)}"
            )

        if df.empty:
            raise IngestionError("Dataset is empty.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise IngestionError("Dataset index must be a DatetimeIndex.")

        if not df.index.is_monotonic_increasing:
            raise IngestionError("Dataset timestamps are not sorted in ascending order.")

        if not df.index.is_unique:
            raise IngestionError("Dataset contains duplicate timestamps.")

        prices = df[["Open", "High", "Low", "Close"]]
        if (prices <= 0).any().any():
            raise IngestionError("Dataset contains non-positive OHLC prices.")

        if (df["High"] < df[["Open", "Close"]].max(axis=1)).any():
            raise IngestionError("Dataset contains rows where High is below Open/Close.")

        if (df["Low"] > df[["Open", "Close"]].min(axis=1)).any():
            raise IngestionError("Dataset contains rows where Low is above Open/Close.")

        request_start_ts = pd.Timestamp(requested_start)
        request_end_ts = pd.Timestamp(requested_end)
        first_ts = pd.Timestamp(df.index.min())
        last_ts = pd.Timestamp(df.index.max())
        if first_ts > request_end_ts or last_ts < request_start_ts:
            raise IngestionError(
                "Dataset timestamps do not overlap the requested time window."
            )

        if str(provenance.get("symbol", "")) != expected_symbol:
            raise DataProvenanceError("Provenance symbol does not match requested symbol.")
        if str(provenance.get("timeframe", "")) != expected_timeframe:
            raise DataProvenanceError("Provenance timeframe does not match requested timeframe.")

        provenance_source = str(provenance.get("source", "")).lower()
        if self.strict_research_mode and provenance_source != expected_source.lower():
            if not (allow_fallback and provenance_source == "fallback"):
                raise DataProvenanceError(
                    f"Research mode requires source '{expected_source}', got '{provenance_source}'."
                )
        if self.strict_research_mode and bool(provenance.get("fallback_used")) and not allow_fallback:
            raise DataProvenanceError(
                "Research mode forbids fallback/synthetic data."
            )

    def _fetch_from_mt5(
        self,
        *,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        mt5 = self._load_mt5_module()
        if mt5 is None:
            raise IngestionError(
                "MetaTrader5 package is not installed in the active Python environment. "
                "Install MetaTrader5 and ensure the MT5 terminal is available."
            )

        logger.info("Downloading %s data from MetaTrader5.", ticker)
        if not mt5.initialize():
            error = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
            raise IngestionError(
                f"MetaTrader5 initialization failed. Last error: {error}. "
                "Open MT5 terminal, log in to your broker account, and retry."
            )

        try:
            symbol_info = mt5.symbol_info(ticker) if hasattr(mt5, "symbol_info") else None
            if symbol_info is None:
                raise IngestionError(
                    f"MT5 symbol '{ticker}' is not available in Market Watch for this terminal/account."
                )

            if not mt5.symbol_select(ticker, True):
                error = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
                raise IngestionError(
                    f"Unable to select symbol '{ticker}' in MetaTrader5. Last error: {error}."
                )

            timeframe = self._map_interval_to_mt5(interval, mt5)
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            df, retrieval_metadata = self._fetch_mt5_rates(
                mt5, ticker, timeframe, interval, start_dt, end_dt
            )

            if df.empty:
                raise IngestionError(
                    f"MetaTrader5 returned zero rows for '{ticker}' ({interval}) between {start_date} and {end_date}."
                )
            if "time" not in df.columns:
                raise IngestionError(
                    "MetaTrader5 response is missing required 'time' column."
                )

            required_mt5_cols = ["time", "open", "high", "low", "close", "tick_volume"]
            missing = [column for column in required_mt5_cols if column not in df.columns]
            if missing:
                raise IngestionError(
                    "MetaTrader5 response is missing required columns: " + ", ".join(missing)
                )

            df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df["Date"] = pd.to_datetime(df["Date"], unit="s")
            df = df.set_index("Date")
            df.index.name = "Date"
            terminal_info = mt5.terminal_info() if hasattr(mt5, "terminal_info") else None
            account_info = mt5.account_info() if hasattr(mt5, "account_info") else None
            retrieval_metadata.update(
                {
                    "mt5_timeframe": self._mt5_timeframe_name(interval),
                    "broker": getattr(account_info, "company", None),
                    "server": getattr(account_info, "server", None),
                    "terminal": getattr(terminal_info, "name", None),
                }
            )
            df.attrs["mt5_retrieval"] = retrieval_metadata
            return df
        finally:
            mt5.shutdown()

    def _normalize_date(self, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.lower() in {"current", "today", "now"}:
                return datetime.now().strftime("%Y-%m-%d")
            return candidate
        return value

    def _build_fallback_dataframe(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        freq = interval.replace("m", "min") if interval.endswith("m") else interval.replace("h", "h")
        dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        if len(dates) == 0:
            dates = pd.date_range(start=start_dt, periods=50, freq="15min")

        base_price = 100.0
        values = []
        prev_close = base_price
        for idx, date in enumerate(dates):
            if idx == 0:
                open_price = prev_close
            else:
                swing = 0.8 if idx % 3 == 0 else -0.6
                if idx % 7 == 0:
                    swing += 0.4
                if idx % 11 == 0:
                    swing -= 0.6
                open_price = prev_close + swing
                open_price = max(open_price, 0.5)

            close_shift = 0.5 if idx % 2 == 0 else -0.4
            if idx % 5 == 0:
                close_shift += 0.3
            if idx % 13 == 0:
                close_shift -= 0.7
            close_price = open_price + close_shift
            close_price = max(close_price, 1.0)
            high_price = max(open_price, close_price) + 0.45
            low_price = max(min(open_price, close_price) - 0.45, 0.5)
            volume = 1000 + idx * 5
            values.append((open_price, high_price, low_price, close_price, volume))
            prev_close = close_price

        df = pd.DataFrame(values, index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
        df.index.name = "Date"
        logger.info("Using built-in fallback sample data for %s (%s -> %s).", ticker, start_date, end_date)
        return df

    def _load_mt5_module(self):
        if "MetaTrader5" in sys.modules:
            return sys.modules["MetaTrader5"]
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            return None
        return mt5

    def _fetch_mt5_rates(
        self,
        mt5,
        ticker: str,
        timeframe: int,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not hasattr(mt5, "copy_rates_range"):
            raise IngestionError("MetaTrader5 does not expose copy_rates_range().")

        bar_delta = self._interval_timedelta(interval)
        chunk_span = bar_delta * (self.MT5_SAFE_CHUNK_BARS - 1)
        chunk_start = self._as_utc(start_dt)
        requested_end = self._as_utc(end_dt)
        chunks: list[pd.DataFrame] = []

        while chunk_start <= requested_end:
            chunk_end = min(chunk_start + chunk_span, requested_end)
            rates = mt5.copy_rates_range(ticker, timeframe, chunk_start, chunk_end)
            last_error = mt5.last_error()
            if rates is None or len(rates) == 0:
                raise IngestionError(
                    f"MetaTrader5 returned no rates for '{ticker}' ({interval}) chunk "
                    f"{chunk_start.isoformat()} to {chunk_end.isoformat()}. "
                    f"Last MT5 error: {last_error}."
                )

            chunk = pd.DataFrame(rates)
            self._validate_mt5_chunk(chunk, ticker, interval, chunk_start, chunk_end)
            chunks.append(chunk)

            if chunk_end >= requested_end:
                break
            chunk_start = chunk_end - bar_delta

        combined = pd.concat(chunks, ignore_index=True)
        combined.sort_values("time", inplace=True)
        duplicate_count = int(combined.duplicated(subset=["time"]).sum())
        combined.drop_duplicates(subset=["time"], keep="last", inplace=True)

        start_epoch = int(self._as_utc(start_dt).timestamp())
        end_epoch = int(requested_end.timestamp())
        combined = combined.loc[
            (combined["time"] >= start_epoch) & (combined["time"] <= end_epoch)
        ].copy()
        if combined.empty:
            raise IngestionError(
                f"MetaTrader5 returned no rows inside requested range for '{ticker}' "
                f"({start_dt.date()} to {end_dt.date()})."
            )

        actual_start = datetime.fromtimestamp(int(combined["time"].iloc[0]), tz=timezone.utc)
        actual_end = datetime.fromtimestamp(int(combined["time"].iloc[-1]), tz=timezone.utc)
        if actual_start > self._as_utc(start_dt) + self.MT5_COVERAGE_TOLERANCE:
            raise IngestionError(
                f"Insufficient MT5 history for '{ticker}' ({interval}): requested start "
                f"{self._as_utc(start_dt).isoformat()}, first available bar "
                f"{actual_start.isoformat()}."
            )
        if actual_end < requested_end - self.MT5_COVERAGE_TOLERANCE:
            raise IngestionError(
                f"Insufficient MT5 history for '{ticker}' ({interval}): requested end "
                f"{requested_end.isoformat()}, last available bar {actual_end.isoformat()}."
            )

        return combined, {
            "chunk_count": len(chunks),
            "duplicate_count": duplicate_count,
        }

    def _validate_mt5_chunk(
        self,
        chunk: pd.DataFrame,
        ticker: str,
        interval: str,
        chunk_start: datetime,
        chunk_end: datetime,
    ) -> None:
        required = {"time", "open", "high", "low", "close", "tick_volume"}
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise IngestionError(
                f"MetaTrader5 response for '{ticker}' ({interval}) chunk "
                f"{chunk_start.isoformat()} to {chunk_end.isoformat()} is missing columns: "
                f"{', '.join(missing)}."
            )

        prices = chunk[["open", "high", "low", "close"]]
        if prices.isna().any().any() or (prices <= 0).any().any():
            raise IngestionError("MetaTrader5 chunk contains invalid OHLC prices.")
        if (chunk["high"] < chunk[["open", "close"]].max(axis=1)).any():
            raise IngestionError("MetaTrader5 chunk contains invalid High values.")
        if (chunk["low"] > chunk[["open", "close"]].min(axis=1)).any():
            raise IngestionError("MetaTrader5 chunk contains invalid Low values.")

    def _as_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _interval_timedelta(self, interval: str) -> timedelta:
        mapping = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        result = mapping.get(interval)
        if result is None:
            raise IngestionError(f"Unsupported interval '{interval}' for MT5 chunking.")
        return result

    def _mt5_timeframe_name(self, interval: str) -> str:
        return {
            "1m": "M1",
            "5m": "M5",
            "15m": "M15",
            "30m": "M30",
            "1h": "H1",
            "4h": "H4",
            "1d": "D1",
        }[interval]

    def _map_interval_to_mt5(self, interval: str, mt5) -> int:
        mapping = {
            "1m": getattr(mt5, "TIMEFRAME_M1", None),
            "5m": getattr(mt5, "TIMEFRAME_M5", None),
            "15m": getattr(mt5, "TIMEFRAME_M15", None),
            "30m": getattr(mt5, "TIMEFRAME_M30", None),
            "1h": getattr(mt5, "TIMEFRAME_H1", None),
            "4h": getattr(mt5, "TIMEFRAME_H4", None),
            "1d": getattr(mt5, "TIMEFRAME_D1", None),
        }
        timeframe = mapping.get(interval)
        if timeframe is None:
            raise IngestionError(f"Unsupported interval '{interval}' for MetaTrader5.")
        return timeframe

    def _bars_per_day(self, interval: str) -> int:
        mapping = {
            "1m": 1440,
            "5m": 288,
            "15m": 96,
            "30m": 48,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }
        return mapping.get(interval, 96)

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load OHLCV data from a local CSV file.

        The CSV must contain at minimum the columns
        ``Date, Open, High, Low, Close, Volume``.
        """
        df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        return df
