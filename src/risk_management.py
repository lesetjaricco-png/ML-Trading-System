"""Risk management: position sizing and trade-level stop-loss / take-profit."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single completed trade."""

    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    direction: int = 1  # 1 = long, -1 = short (long-only for now)
    quantity: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    entry_fill_price: float = 0.0
    exit_fill_price: float = 0.0
    exit_reason: str = ""


class RiskManager:
    """Enforce position sizing and per-trade risk rules.

    Parameters
    ----------
    max_position_size:
        Maximum fraction of portfolio to allocate to a single position.
    stop_loss:
        Fraction below entry price that triggers a stop-loss exit.
    take_profit:
        Fraction above entry price that triggers a take-profit exit.
    max_drawdown:
        If portfolio drawdown exceeds this fraction, no new trades are opened.
    risk_per_trade:
        Fraction of portfolio to risk on each trade (Kelly-inspired sizing).
    """

    def __init__(
        self,
        max_position_size: float = 0.1,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        max_drawdown: float = 0.15,
        risk_per_trade: float = 0.01,
    ):
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        atr: Optional[float] = None,
        instrument_spec: Optional[object] = None,
        stop_loss_points: int = 20,
    ) -> float:
        """Return the position size in contract/lot units.

        When an instrument specification is provided, the size is derived from
        the configured risk-per-trade fraction and the monetary loss of the
        configured stop-loss distance. The result is capped by the configured
        maximum position size.
        """
        if portfolio_value <= 0 or entry_price <= 0:
            return 0.0

        if instrument_spec is not None:
            risk_amount = portfolio_value * self.risk_per_trade
            point_size = getattr(instrument_spec, "effective_point_size", lambda: getattr(instrument_spec, "point_size", 0.01))()
            point_size = float(point_size or 0.01)
            stop_distance = max(float(stop_loss_points) * point_size, 1e-9)
            tick_value = getattr(instrument_spec, "tick_value", None)
            contract_size = getattr(instrument_spec, "contract_size", None)
            monetary_loss_per_unit = None
            if tick_value is not None and tick_value > 0:
                monetary_loss_per_unit = stop_distance * float(tick_value)
            elif contract_size is not None and contract_size > 0:
                monetary_loss_per_unit = stop_distance * float(contract_size)
            else:
                monetary_loss_per_unit = stop_distance * float(entry_price)

            if monetary_loss_per_unit is None or monetary_loss_per_unit <= 0:
                return 0.0

            raw_size = risk_amount / monetary_loss_per_unit
            size = float(raw_size)
            max_allowed = float(self.max_position_size)
            if max_allowed <= 0:
                return 0.0

            volume_min = getattr(instrument_spec, "volume_min", None)
            volume_step = getattr(instrument_spec, "volume_step", None)
            if volume_min is not None and volume_min > 0 and size < volume_min:
                size = float(volume_min)
            if volume_step is not None and volume_step > 0:
                size = math.floor(size / float(volume_step) + 1e-12) * float(volume_step)

            volume_max = getattr(instrument_spec, "volume_max", None)
            if volume_max is not None and volume_max > 0:
                max_allowed = min(max_allowed, float(volume_max))
            return max(0.0, min(size, max_allowed))

        if atr and atr > 0:
            risk_amount = portfolio_value * self.risk_per_trade
            stop_distance = atr * 2
            raw_shares = risk_amount / stop_distance
        else:
            allocated_capital = portfolio_value * self.max_position_size
            raw_shares = allocated_capital / entry_price

        max_shares = (portfolio_value * self.max_position_size) / entry_price
        shares = min(raw_shares, max_shares)
        return max(0.0, float(shares))

    def calculate_stop_loss(self, entry_price: float, atr: Optional[float] = None) -> float:
        """Return the stop-loss price for a long position."""
        if atr and atr > 0:
            return entry_price - 2 * atr
        return entry_price * (1 - self.stop_loss)

    def calculate_take_profit(self, entry_price: float, atr: Optional[float] = None) -> float:
        """Return the take-profit price for a long position."""
        if atr and atr > 0:
            return entry_price + 3 * atr
        return entry_price * (1 + self.take_profit)

    def check_drawdown_limit(
        self, portfolio_value: float, peak_value: float
    ) -> bool:
        """Return *True* if it is safe to open new positions."""
        drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        if drawdown >= self.max_drawdown:
            logger.warning(
                "Max drawdown reached (%.2f%%). No new trades.", drawdown * 100
            )
            return False
        return True

    def should_exit(
        self,
        trade: Trade,
        current_price: float,
    ) -> tuple[bool, str]:
        """Return ``(should_exit, reason)`` for an open long trade."""
        if current_price <= trade.stop_loss_price:
            return True, "stop_loss"
        if current_price >= trade.take_profit_price:
            return True, "take_profit"
        return False, ""
