"""Feature engineering: derive technical indicators from OHLCV data."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute technical analysis features and generate ML target labels.

    Parameters
    ----------
    rsi_period:
        Look-back window for the Relative Strength Index.
    macd_fast / macd_slow / macd_signal:
        EMA periods for MACD computation.
    bb_period / bb_std:
        Bollinger Bands window and number of standard deviations.
    sma_periods:
        List of Simple Moving Average windows.
    ema_periods:
        List of Exponential Moving Average windows.
    atr_period:
        Look-back window for the Average True Range.
    volume_sma_period:
        Window for volume Simple Moving Average.
    lookahead_days:
        Number of future bars used to compute the target label.
    return_threshold:
        Minimum forward return required to label a bar as a buy signal.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        sma_periods: List[int] = None,
        ema_periods: List[int] = None,
        atr_period: int = 14,
        volume_sma_period: int = 20,
        lookahead_days: int = 5,
        return_threshold: float = 0.01,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_periods = sma_periods or [10, 20, 50, 200]
        self.ema_periods = ema_periods or [12, 26]
        self.atr_period = atr_period
        self.volume_sma_period = volume_sma_period
        self.lookahead_days = lookahead_days
        self.return_threshold = return_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features and the binary target column to *df*.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame with columns ``Open, High, Low, Close, Volume``.

        Returns
        -------
        pd.DataFrame
            Original data plus feature columns and a ``target`` column.
            Rows with NaN values (due to look-back windows) are dropped.
        """
        df = df.copy()
        df = self._add_price_features(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_moving_averages(df)
        df = self._add_atr(df)
        df = self._add_volume_features(df)
        df = self._add_candlestick_features(df)
        df = self._add_target(df)
        df.dropna(inplace=True)
        logger.info("Feature matrix shape after engineering: %s", df.shape)
        return df

    @property
    def feature_columns(self) -> List[str]:
        """Return the list of feature column names (excludes OHLCV + target)."""
        return [
            "returns",
            "log_returns",
            "high_low_ratio",
            "close_open_ratio",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_pct",
            *[f"sma_{p}" for p in self.sma_periods],
            *[f"ema_{p}" for p in self.ema_periods],
            *[f"price_to_sma_{p}" for p in self.sma_periods],
            "atr",
            "atr_pct",
            "volume_sma",
            "volume_ratio",
            "obv",
            "body_size",
            "upper_shadow",
            "lower_shadow",
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["high_low_ratio"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_open_ratio"] = (df["Close"] - df["Open"]) / df["Open"]
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["Close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = df["Close"].rolling(self.bb_period).mean()
        std = df["Close"].rolling(self.bb_period).std()
        df["bb_upper"] = sma + self.bb_std * std
        df["bb_lower"] = sma - self.bb_std * std
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_width"] = bb_range / sma
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.sma_periods:
            df[f"sma_{period}"] = df["Close"].rolling(period).mean()
            df[f"price_to_sma_{period}"] = df["Close"] / df[f"sma_{period}"]
        for period in self.ema_periods:
            df[f"ema_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["Close"]
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["volume_sma"] = df["Volume"].rolling(self.volume_sma_period).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma"].replace(0, np.nan)
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
        df["obv"] = obv
        return df

    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["body_size"] = (df["Close"] - df["Open"]).abs() / df["Open"]
        df["upper_shadow"] = (df["High"] - df[["Close", "Open"]].max(axis=1)) / df["Open"]
        df["lower_shadow"] = (df[["Close", "Open"]].min(axis=1) - df["Low"]) / df["Open"]
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary label: 1 if forward return >= threshold, else 0."""
        forward_return = df["Close"].shift(-self.lookahead_days) / df["Close"] - 1
        df["target"] = (forward_return >= self.return_threshold).astype(int)
        return df
