"""Data ingestion module: fetch and cache OHLCV market data."""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataIngestion:
    """Downloads and caches historical OHLCV data using yfinance."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
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
            self.data_dir, f"{ticker}_{start_date}_{end_date}_{interval}.parquet"
        )

        if use_cache and os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_parquet(cache_path)
            return df

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
