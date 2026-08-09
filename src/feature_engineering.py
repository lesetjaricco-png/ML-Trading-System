"""Feature engineering: derive technical indicators from OHLCV data."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.instruments import InstrumentSpec, resolve_instrument_spec

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
    timeframe:
        Candle size of the raw market data (for example "15m").
    take_profit_points:
        Price distance in points to define the take-profit level.
    stop_loss_points:
        Price distance in points to define the stop-loss level.
    max_bars:
        Maximum number of future candles to examine before declaring the trade unresolved.
    same_bar_rule:
        How to handle cases where both TP and SL would be reached in the same bar.
    unresolved_policy:
        What to do when neither TP nor SL is reached within max_bars.
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
        timeframe: str = "15m",
        take_profit_points: float = 100.0,
        stop_loss_points: float = 20.0,
        max_bars: int = 40,
        same_bar_rule: str = "drop",
        unresolved_policy: str = "drop",
        instrument_config: Dict[str, Dict[str, Any]] | None = None,
        instrument_spec: InstrumentSpec | None = None,
        target_mode: str = "v0.1_tp_before_sl",
        forward_horizon: int = 5,
        atr_threshold_multiplier: float = 1.0,
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
        self.timeframe = timeframe
        self.take_profit_points = take_profit_points
        self.stop_loss_points = stop_loss_points
        self.max_bars = max_bars
        self.same_bar_rule = same_bar_rule.lower()
        self.unresolved_policy = unresolved_policy.lower()
        self.instrument_config = instrument_config or {}
        self.instrument_spec = instrument_spec
        self.target_mode = target_mode.lower()
        self.forward_horizon = forward_horizon
        self.atr_threshold_multiplier = atr_threshold_multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame, instrument_name: str | None = None) -> pd.DataFrame:
        """Add causal features and the configured target column to *df*.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame with columns ``Open, High, Low, Close, Volume``.
        instrument_name:
            Instrument identifier used to resolve per-instrument TP/SL overrides.

        Returns
        -------
        pd.DataFrame
            Original data plus feature columns and a ``target`` column.
            Rows with NaN values (due to look-back windows) are dropped.
        """
        df = df.copy()
        if df.empty:
            return df

        df = df.sort_index()
        if df.index.duplicated().any():
            raise ValueError("Input data contains duplicate timestamps; feature generation requires unique chronological rows.")

        df = self._add_price_features(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_moving_averages(df)
        df = self._add_atr(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_candlestick_features(df)
        df = self._add_time_features(df)
        df = self._add_target(df, instrument_name=instrument_name)

        feature_columns = [col for col in self.feature_columns if col != "target"]
        available_feature_columns = [column for column in feature_columns if column in df.columns]
        assert "target" not in available_feature_columns
        assert not any(column.startswith("future_") or column.endswith("_outcome") for column in available_feature_columns)
        df = df.dropna(subset=["target"])
        if available_feature_columns:
            df = df.dropna(subset=available_feature_columns)
        if df.empty:
            logger.info("Feature matrix empty after dropping warm-up rows and NaN features; returning empty dataframe")
            return df
        logger.info("Feature matrix shape after engineering: %s", df.shape)
        return df

    @property
    def feature_columns(self) -> List[str]:
        """Return the list of feature column names (excludes OHLCV + target)."""
        return [
            "returns",
            "returns_2",
            "returns_5",
            "returns_10",
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
            "volatility_5",
            "volatility_20",
            "volume_sma",
            "volume_ratio",
            "obv",
            "body_size",
            "upper_shadow",
            "lower_shadow",
            "day_of_week",
            "is_weekend",
            "hour_of_day",
            "is_market_open",
            "is_asia_session",
            "is_london_session",
            "is_new_york_session",
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["returns"] = df["Close"].pct_change()
        for period in [2, 5, 10]:
            df[f"returns_{period}"] = df["Close"].pct_change(period)
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["high_low_ratio"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_open_ratio"] = (df["Close"] - df["Open"]) / df["Open"]
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["Close"].diff().fillna(0.0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)
        df["rsi"] = rsi
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

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["Close"].pct_change()
        df["volatility_5"] = returns.rolling(5).std()
        df["volatility_20"] = returns.rolling(20).std()
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

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["hour_of_day"] = df.index.hour
        df["is_market_open"] = 1
        df["is_asia_session"] = df["hour_of_day"].between(0, 7).astype(int)
        df["is_london_session"] = df["hour_of_day"].between(8, 15).astype(int)
        df["is_new_york_session"] = df["hour_of_day"].between(13, 20).astype(int)
        return df

    def _add_target(self, df: pd.DataFrame, instrument_name: str | None = None) -> pd.DataFrame:
        """Create the configured TP/SL, directional, or forward-ATR target."""
        df = df.copy()
        if self.target_mode == "v0.2_directional":
            df["target"] = self._build_directional_target_series(df, instrument_name=instrument_name)
        elif self.target_mode == "v0.3_forward_atr":
            df["target"] = self._build_v03_forward_atr_target_series(df, instrument_name=instrument_name)
        else:
            df["target"] = self._build_target_series(df, instrument_name=instrument_name)
        return df

    def _build_target_series(self, df: pd.DataFrame, instrument_name: str | None = None) -> List[float]:
        target = []
        for idx in range(len(df)):
            entry_price = df.iloc[idx]["Close"]
            tp_price, sl_price = self._resolve_tp_sl_levels(entry_price, instrument_name)
            outcome = self._evaluate_trade_outcome(df, idx, tp_price, sl_price)
            target.append(outcome)
        return target

    def _build_directional_target_series(self, df: pd.DataFrame, instrument_name: str | None = None) -> List[float]:
        target = []
        for idx in range(len(df)):
            entry_price = df.iloc[idx]["Close"]
            tp_price, sl_price = self._resolve_tp_sl_levels(entry_price, instrument_name)
            long_outcome = self._evaluate_trade_outcome(df, idx, tp_price, sl_price)
            short_tp_price = entry_price - self.take_profit_points * self._resolve_point_size(instrument_name)
            short_sl_price = entry_price + self.stop_loss_points * self._resolve_point_size(instrument_name)
            short_outcome = self._evaluate_trade_outcome_for_direction(df, idx, short_tp_price, short_sl_price)
            target.append(
                self._classify_directional_target(
                    df=df,
                    entry_idx=idx,
                    entry_price=entry_price,
                    long_outcome=long_outcome,
                    short_outcome=short_outcome,
                    instrument_name=instrument_name,
                )
            )
        return target

    def _build_v03_forward_atr_target_series(self, df: pd.DataFrame, instrument_name: str | None = None) -> List[float]:
        """Label the full h-bar forward close return against ATR% known at t.

        Classes are 1 (above +threshold), 0 (below -threshold), and 2
        (inside the threshold). Rows without the complete future horizon are
        unlabeled and excluded by ``transform``.
        """
        target = []
        for idx in range(len(df)):
            entry_row = df.iloc[idx]
            entry_close = float(entry_row["Close"])
            if entry_close == 0:
                target.append(2)
                continue

            if idx + self.forward_horizon >= len(df):
                target.append(np.nan)
                continue

            future_row = df.iloc[idx + self.forward_horizon]
            future_close = float(future_row["Close"])
            forward_return = (future_close / entry_close) - 1.0
            atr_pct = float(entry_row.get("atr_pct", np.nan))
            threshold = abs(self.atr_threshold_multiplier * atr_pct) if np.isfinite(atr_pct) else 0.0
            if forward_return > threshold:
                target.append(1)
            elif forward_return < -threshold:
                target.append(0)
            else:
                target.append(2)
        return target

    def _classify_directional_target(
        self,
        *,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        long_outcome: int | float,
        short_outcome: int | float,
        instrument_name: str | None = None,
    ) -> int:
        if long_outcome == 1 and short_outcome == 0:
            return 1
        if long_outcome == 0 and short_outcome == 1:
            return 0

        horizon = min(self.max_bars, len(df) - entry_idx - 1)
        if horizon <= 0:
            return 2

        future_prices = df.iloc[entry_idx + 1 : entry_idx + 1 + horizon]["Close"]
        if future_prices.empty:
            return 2

        last_future_close = float(future_prices.iloc[-1])
        point_size = self._resolve_point_size(instrument_name)
        tp_distance = self.take_profit_points * point_size
        sl_distance = self.stop_loss_points * point_size

        if last_future_close >= entry_price + tp_distance:
            return 1
        if last_future_close <= entry_price - sl_distance:
            return 0

        cumulative_return = (last_future_close / entry_price) - 1.0
        if cumulative_return > 0.0:
            return 1
        if cumulative_return < 0.0:
            return 0
        return 2

    def _evaluate_trade_outcome_for_direction(self, df: pd.DataFrame, entry_idx: int, tp_price: float, sl_price: float) -> int | float:
        for offset in range(1, self.max_bars + 1):
            if entry_idx + offset >= len(df):
                break
            row = df.iloc[entry_idx + offset]
            high = row["High"]
            low = row["Low"]
            if high >= tp_price and low <= sl_price:
                if self.same_bar_rule in {"tp_first", "take_profit_first"}:
                    return 1
                if self.same_bar_rule in {"sl_first", "conservative_sl", "stop_loss_first"}:
                    return 0
                if self.same_bar_rule == "drop":
                    return float("nan")
                raise ValueError(f"Unsupported same_bar_rule: {self.same_bar_rule}")
            if high >= tp_price:
                return 1
            if low <= sl_price:
                return 0
        if self.unresolved_policy == "drop":
            return float("nan")
        if self.unresolved_policy == "sl":
            return 0
        if self.unresolved_policy == "tp":
            return 1
        raise ValueError(f"Unsupported unresolved_policy: {self.unresolved_policy}")

    def _resolve_tp_sl_levels(self, entry_price: float, instrument_name: str | None = None) -> tuple[float, float]:
        config = self._resolve_target_config(instrument_name)
        tp_points = config.get("take_profit_points", self.take_profit_points)
        sl_points = config.get("stop_loss_points", self.stop_loss_points)
        point_size = self._resolve_point_size(instrument_name)
        return entry_price + tp_points * point_size, entry_price - sl_points * point_size

    def _resolve_point_size(self, instrument_name: str | None = None) -> float:
        config = self._resolve_target_config(instrument_name)
        configured_point_size = config.get("point_size")
        if configured_point_size is not None:
            return float(configured_point_size)

        if self.instrument_spec is not None:
            return self.instrument_spec.effective_point_size()

        if instrument_name:
            resolved_spec = resolve_instrument_spec(instrument_name, fallback_point_size=0.01)
            return resolved_spec.effective_point_size()

        return 0.01

    def _resolve_target_config(self, instrument_name: str | None = None) -> Dict[str, Any]:
        if instrument_name and instrument_name in self.instrument_config:
            return self.instrument_config[instrument_name]
        return {}

    def evaluate_target_summary(
        self,
        df: pd.DataFrame,
        instrument_name: str | None = None,
        instrument_config: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate the TP-before-SL target independently of XGBoost."""
        if instrument_config is not None:
            self.instrument_config = instrument_config

        if self.target_mode == "v0.2_directional":
            target = self._build_directional_target_series(df, instrument_name)
        elif self.target_mode == "v0.3_forward_atr":
            target = self._build_v03_forward_atr_target_series(df, instrument_name)
        else:
            target = self._build_target_series(df, instrument_name)
        valid = [value for value in target if pd.notna(value)]
        wins = sum(1 for value in valid if value == 1)
        losses = sum(1 for value in valid if value == 0)
        dropped = sum(1 for value in target if pd.isna(value))

        total_observations = len(df)
        win_rate = wins / len(valid) if valid else 0.0
        total_points = sum(
            self._trade_pnl_points(value) for value in valid
        )
        avg_win = sum(
            self._trade_pnl_points(value) for value in valid if value == 1
        ) / wins if wins else 0.0
        avg_loss = sum(
            self._trade_pnl_points(value) for value in valid if value == 0
        ) / losses if losses else 0.0
        gross_profit = sum(self._trade_pnl_points(value) for value in valid if value == 1)
        gross_loss = abs(sum(self._trade_pnl_points(value) for value in valid if value == 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        monthly_breakdown = []
        if isinstance(df.index, pd.DatetimeIndex):
            monthly_groups = df.groupby(df.index.to_period("M"))
            for period, group in monthly_groups:
                if self.target_mode == "v0.2_directional":
                    group_target = self._build_directional_target_series(group, instrument_name)
                elif self.target_mode == "v0.3_forward_atr":
                    group_target = self._build_v03_forward_atr_target_series(group, instrument_name)
                else:
                    group_target = self._build_target_series(group, instrument_name)
                valid_group = [value for value in group_target if pd.notna(value)]
                wins_group = sum(1 for value in valid_group if value == 1)
                losses_group = sum(1 for value in valid_group if value == 0)
                dropped_group = sum(1 for value in group_target if pd.isna(value))
                monthly_breakdown.append(
                    {
                        "period": str(period),
                        "trades": len(valid_group),
                        "wins": wins_group,
                        "losses": losses_group,
                        "dropped": dropped_group,
                        "win_rate": wins_group / len(valid_group) if valid_group else 0.0,
                        "pnl_points": sum(self._trade_pnl_points(value) for value in valid_group),
                    }
                )

        return {
            "total_observations": total_observations,
            "total_trades": len(valid),
            "wins": wins,
            "losses": losses,
            "dropped": dropped,
            "win_rate": round(win_rate, 4),
            "total_pnl_points": round(total_points, 2),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(self._calculate_max_drawdown(target), 2),
            "monthly_breakdown": monthly_breakdown,
        }

    def _trade_pnl_points(self, value: float) -> float:
        if value == 1:
            return 1.0
        if value == 0:
            return -1.0
        return 0.0

    def _calculate_max_drawdown(self, target: List[float]) -> float:
        running = []
        cumulative = 0.0
        peak = 0.0
        for value in target:
            if pd.isna(value):
                continue
            cumulative += self._trade_pnl_points(value)
            running.append(cumulative)
            if cumulative > peak:
                peak = cumulative
        if not running:
            return 0.0
        return max(0.0, peak - min(running))

    def _evaluate_trade_outcome(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        tp_price: float,
        sl_price: float,
    ) -> int | float:
        for offset in range(1, self.max_bars + 1):
            if entry_idx + offset >= len(df):
                break

            row = df.iloc[entry_idx + offset]
            high = row["High"]
            low = row["Low"]

            if high >= tp_price and low <= sl_price:
                if self.same_bar_rule in {"tp_first", "take_profit_first"}:
                    return 1
                if self.same_bar_rule in {"sl_first", "conservative_sl", "stop_loss_first"}:
                    return 0
                if self.same_bar_rule == "drop":
                    return float("nan")
                raise ValueError(f"Unsupported same_bar_rule: {self.same_bar_rule}")

            if high >= tp_price:
                return 1
            if low <= sl_price:
                return 0

        if self.unresolved_policy == "drop":
            return float("nan")
        if self.unresolved_policy == "sl":
            return 0
        if self.unresolved_policy == "tp":
            return 1
        raise ValueError(f"Unsupported unresolved_policy: {self.unresolved_policy}")
