"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import BacktestEngine
from src.risk_management import RiskManager


@pytest.fixture
def signal_df():
    """A simple DataFrame with pre-set signals for deterministic testing."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(1)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    close = np.clip(close, 1, None)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.1
    atr = np.full(n, 1.0)
    # Generate BUY signals every 20 bars
    signal = np.zeros(n, dtype=int)
    signal[::20] = 1
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "atr": atr,
            "signal": signal,
        },
        index=dates,
    )


class TestBacktestEngine:
    def test_run_returns_dataframe(self, signal_df):
        engine = BacktestEngine(initial_capital=100_000)
        equity = engine.run(signal_df)
        assert isinstance(equity, pd.DataFrame)
        assert "portfolio_value" in equity.columns

    def test_equity_length_matches_input(self, signal_df):
        engine = BacktestEngine(initial_capital=100_000)
        equity = engine.run(signal_df)
        assert len(equity) == len(signal_df)

    def test_initial_portfolio_value(self, signal_df):
        engine = BacktestEngine(initial_capital=100_000)
        equity = engine.run(signal_df)
        # First row: no position opened yet, so portfolio = initial_capital
        assert equity["portfolio_value"].iloc[0] == pytest.approx(100_000, rel=0.05)

    def test_performance_metrics_keys(self, signal_df):
        engine = BacktestEngine(initial_capital=100_000)
        equity = engine.run(signal_df)
        metrics = engine.performance_metrics(equity)
        for key in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"):
            assert key in metrics

    def test_no_cash_goes_negative(self, signal_df):
        engine = BacktestEngine(initial_capital=50_000)
        engine.run(signal_df)
        assert engine.cash >= 0

    def test_max_drawdown_non_negative(self, signal_df):
        engine = BacktestEngine(initial_capital=100_000)
        equity = engine.run(signal_df)
        metrics = engine.performance_metrics(equity)
        assert metrics["max_drawdown"] >= 0

    def test_win_rate_in_range(self, signal_df):
        engine = BacktestEngine(initial_capital=100_000)
        equity = engine.run(signal_df)
        metrics = engine.performance_metrics(equity)
        assert 0 <= metrics["win_rate"] <= 100

    def test_no_signals_no_trades(self, signal_df):
        signal_df = signal_df.copy()
        signal_df["signal"] = 0
        engine = BacktestEngine(initial_capital=100_000)
        engine.run(signal_df)
        assert engine.trades == []
