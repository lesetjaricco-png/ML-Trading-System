"""Tests for the risk management module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk_management import RiskManager, Trade


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            max_position_size=0.1,
            stop_loss=0.02,
            take_profit=0.04,
            max_drawdown=0.15,
            risk_per_trade=0.01,
        )

    def test_position_size_flat(self):
        qty = self.rm.calculate_position_size(100_000, 100.0)
        assert qty > 0
        # max 10 % of 100k → max 100 shares
        assert qty <= 100

    def test_position_size_atr(self):
        qty = self.rm.calculate_position_size(100_000, 100.0, atr=1.0)
        assert qty > 0

    def test_position_size_no_over_allocation(self):
        qty = self.rm.calculate_position_size(1_000, 200.0)
        cost = qty * 200.0
        assert cost <= 1_000 * 0.1 + 1  # within max_position_size (+1 rounding)

    def test_stop_loss_price(self):
        sl = self.rm.calculate_stop_loss(100.0)
        assert sl == pytest.approx(98.0)

    def test_take_profit_price(self):
        tp = self.rm.calculate_take_profit(100.0)
        assert tp == pytest.approx(104.0)

    def test_stop_loss_atr(self):
        sl = self.rm.calculate_stop_loss(100.0, atr=1.0)
        assert sl == pytest.approx(98.0)

    def test_take_profit_atr(self):
        tp = self.rm.calculate_take_profit(100.0, atr=1.0)
        assert tp == pytest.approx(103.0)

    def test_drawdown_limit_ok(self):
        assert self.rm.check_drawdown_limit(90_000, 100_000) is True

    def test_drawdown_limit_exceeded(self):
        assert self.rm.check_drawdown_limit(84_000, 100_000) is False

    def test_should_exit_stop_loss(self):
        trade = Trade(
            entry_date=pd.Timestamp("2023-01-01"),
            entry_price=100.0,
            stop_loss_price=98.0,
            take_profit_price=104.0,
        )
        should_exit, reason = self.rm.should_exit(trade, current_price=97.0)
        assert should_exit is True
        assert reason == "stop_loss"

    def test_should_exit_take_profit(self):
        trade = Trade(
            entry_date=pd.Timestamp("2023-01-01"),
            entry_price=100.0,
            stop_loss_price=98.0,
            take_profit_price=104.0,
        )
        should_exit, reason = self.rm.should_exit(trade, current_price=105.0)
        assert should_exit is True
        assert reason == "take_profit"

    def test_should_not_exit(self):
        trade = Trade(
            entry_date=pd.Timestamp("2023-01-01"),
            entry_price=100.0,
            stop_loss_price=98.0,
            take_profit_price=104.0,
        )
        should_exit, _ = self.rm.should_exit(trade, current_price=101.0)
        assert should_exit is False
