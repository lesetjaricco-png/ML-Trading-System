"""Data ingestion module: fetch and cache OHLCV market data."""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataIngestion:
    """Downloads and caches historical OHLCV data from MT5/IC Markets or yfinance."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "15m",
        source: str = "mt5",
        use_cache: bool = True,
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
        cache_path = os.path.join(
            self.data_dir, f"{ticker}_{start_date}_{end_date}_{interval}_{source}.parquet"
        )

        if use_cache and os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_parquet(cache_path)
            return df

        if source.lower() == "mt5":
            mt5 = self._load_mt5_module()
            if mt5 is None:
                raise ImportError(
                    "MetaTrader5 is required for MT5 data ingestion. Install it and ensure the MT5 terminal is available."
                )

            logger.info("Downloading %s data from IC Markets via MetaTrader5 …", ticker)
            if not mt5.initialize():
                raise RuntimeError("MetaTrader5 initialization failed. Ensure the terminal is installed and running.")

            try:
                if not mt5.symbol_select(ticker, True):
                    raise ValueError(f"Unable to select symbol '{ticker}' in MetaTrader5.")

                timeframe = self._map_interval_to_mt5(interval, mt5)
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                rates = self._fetch_mt5_rates(mt5, ticker, timeframe, interval, start_dt, end_dt)
                if rates is None:
                    raise ValueError(
                        f"No data returned for ticker '{ticker}' ({start_date} – {end_date}, interval={interval})."
                    )

                df = pd.DataFrame(rates)
                if df.empty:
                    raise ValueError(
                        f"No data returned for ticker '{ticker}' ({start_date} – {end_date}, interval={interval})."
                    )

                if "time" not in df.columns:
                    raise ValueError("MetaTrader5 response did not include a 'time' column.")

                df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
                df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
                df["Date"] = pd.to_datetime(df["Date"], unit="s")
                df = df.set_index("Date")
                df.index.name = "Date"
            finally:
                mt5.shutdown()
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

        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        df.to_parquet(cache_path)
        logger.info("Saved %d rows to %s", len(df), cache_path)
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
    ):
        rates = None

        if hasattr(mt5, "copy_rates_range"):
            rates = mt5.copy_rates_range(ticker, timeframe, start_dt, end_dt)
            if rates is not None and len(rates) > 0:
                return rates

        # Some brokers/servers provide limited history windows. If the direct
        # range query is empty, fetch the most recent bars up to end_dt.
        if hasattr(mt5, "copy_rates_from"):
            bars_per_day = self._bars_per_day(interval=interval)
            span_days = max((end_dt - start_dt).days, 1)
            requested_bars = max(int(math.ceil(span_days * bars_per_day)), 1_000)
            requested_bars = min(requested_bars, 200_000)
            rates = mt5.copy_rates_from(ticker, timeframe, end_dt, requested_bars)
            if rates is not None and len(rates) > 0:
                return rates

        raise ValueError(
            f"MetaTrader5 returned no rates for '{ticker}' (timeframe={timeframe}, {start_dt.date()} to {end_dt.date()}). "
            f"Last MT5 error: {mt5.last_error()}."
        )

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
            raise ValueError(f"Unsupported interval '{interval}' for MetaTrader5.")
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
