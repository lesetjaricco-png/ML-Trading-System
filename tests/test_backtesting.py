"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import BacktestEngine
from src.risk_management import RiskManager, Trade
from src.instruments import InstrumentSpec


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

    def test_tp_conversion_uses_instrument_point_size(self):
        spec = InstrumentSpec(symbol="US30", point_size=0.25, tick_size=0.25, tick_value=1.0, contract_size=1.0)
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=spec, take_profit_points=100, stop_loss_points=20)
        tp_price, sl_price = engine._resolve_tp_sl_prices(100.0, 1)
        assert tp_price == pytest.approx(125.0)
        assert sl_price == pytest.approx(95.0)

    def test_buy_execution_closes_at_configured_tp(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 100.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01), take_profit_points=100, stop_loss_points=20)
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "take_profit"
        assert engine.trades[0].exit_price == pytest.approx(101.0)

    def test_buy_execution_closes_at_configured_sl(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 100.0],
                "Low": [100.0, 99.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01), take_profit_points=100, stop_loss_points=20)
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "stop_loss"
        assert engine.trades[0].exit_price == pytest.approx(99.8)

    def test_sell_execution_closes_at_configured_tp(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 100.0],
                "Low": [100.0, 99.0],
                "Close": [100.0, 99.0],
                "signal": [-1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01), take_profit_points=100, stop_loss_points=20)
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "take_profit"
        assert engine.trades[0].exit_price == pytest.approx(99.0)

    def test_open_trade_without_exit_closes_at_end_of_data(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 100.0],
                "Low": [100.0, 100.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01), take_profit_points=100, stop_loss_points=20)
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "end_of_data"

    def test_end_of_data_close_updates_final_equity_snapshot(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 100.0],
                "Low": [100.0, 100.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
            take_profit_points=100,
            stop_loss_points=20,
        )
        equity = engine.run(df)

        assert equity["cash"].iloc[-1] == pytest.approx(100_000.0 + engine.trades[0].pnl)
        assert equity["position_value"].iloc[-1] == pytest.approx(0.0)
        assert equity["portfolio_value"].iloc[-1] == pytest.approx(100_000.0 + engine.trades[0].pnl)

    def test_same_bar_tp_and_sl_does_not_choose_side(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 99.8],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01), take_profit_points=100, stop_loss_points=20)
        engine._open_trade(df.index[0], df.iloc[0])
        should_exit, reason = engine._check_exit(df.index[1], df.iloc[1])
        assert should_exit is False
        assert reason == "same_bar"

    def test_same_bar_signal_does_not_reopen_after_exit(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 100.0],
                "Close": [100.0, 100.0],
                "signal": [1, 1],
            },
            index=dates,
        )
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01),
            take_profit_points=100,
            stop_loss_points=20,
        )
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.max_simultaneous_positions == 1

    def test_accounting_matches_initial_capital_plus_realized_pnl(self):
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
        )
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 99.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        equity = engine.run(df)
        realized_pnl = sum(trade.net_pnl for trade in engine.trades)
        assert equity["portfolio_value"].iloc[-1] == pytest.approx(100_000 + realized_pnl)

    def test_trade_cash_relationship_matches_trade_pnl(self):
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
        )
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 99.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        assert engine._open_trade(df.index[0], df.iloc[0])
        cash_before_close = engine.cash
        engine._close_trade(df.index[1], 101.0, reason="take_profit")
        trade = engine.trades[-1]
        assert engine.cash == pytest.approx(100_000 + trade.net_pnl)

    def test_same_bar_buy_conflict_uses_stop_loss(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 99.8],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
            same_bar_rule="conservative_sl",
        )
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "stop_loss"

    def test_same_bar_sell_conflict_uses_stop_loss(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 101.0],
                "Low": [100.0, 99.0],
                "Close": [100.0, 100.0],
                "signal": [-1, 0],
            },
            index=dates,
        )
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
            same_bar_rule="conservative_sl",
        )
        engine.run(df)
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "stop_loss"

    def test_end_of_data_close_matches_initial_capital_plus_realized_pnl(self):
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
        )
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 100.0],
                "Low": [100.0, 100.0],
                "Close": [100.0, 100.0],
                "signal": [1, 0],
            },
            index=dates,
        )
        equity = engine.run(df)
        realized_pnl = sum(trade.net_pnl for trade in engine.trades)
        assert equity["portfolio_value"].iloc[-1] == pytest.approx(100_000 + realized_pnl)

    def test_position_size_respects_max_position_fraction(self):
        risk = RiskManager(max_position_size=0.1)
        qty = risk.calculate_position_size(100_000, 100.0)
        assert qty <= 100.0

    def test_trade_pnl_matches_realized_cash_change(self):
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
        )
        entry_date = pd.Timestamp("2024-01-01 00:00:00")
        exit_date = pd.Timestamp("2024-01-01 00:15:00")
        entry_row = pd.Series({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "signal": 1})

        assert engine._open_trade(entry_date, entry_row)
        cash_after_open = engine.cash

        engine._close_trade(exit_date, 101.0, reason="take_profit")
        trade = engine.trades[-1]

        assert engine.cash == pytest.approx(100_000.0 + trade.pnl)
        assert engine.cash - cash_after_open == pytest.approx(trade.pnl + (100_000.0 - cash_after_open))

    def test_trade_pnl_includes_entry_and_exit_slippage(self):
        engine = BacktestEngine(
            initial_capital=100_000,
            instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0),
            risk_manager=RiskManager(max_position_size=1e6, risk_per_trade=1e-6),
            commission=0.001,
            slippage=0.0005,
        )
        entry_date = pd.Timestamp("2024-01-01 00:00:00")
        exit_date = pd.Timestamp("2024-01-01 00:15:00")
        entry_row = pd.Series({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "signal": 1})

        assert engine._open_trade(entry_date, entry_row)
        engine._close_trade(exit_date, 101.0, reason="take_profit")
        trade = engine.trades[-1]

        assert trade.net_pnl == pytest.approx(engine.cash - 100_000.0)
        assert trade.slippage > 0

    def test_trade_pnl_uses_quantity_and_price_move(self):
        engine = BacktestEngine(initial_capital=100_000, instrument_spec=InstrumentSpec(symbol="US30", point_size=0.01, tick_value=1.0))
        pnl = engine._calculate_trade_pnl(100.0, 100.0, 101.0)
        assert pnl == pytest.approx(10000.0)
