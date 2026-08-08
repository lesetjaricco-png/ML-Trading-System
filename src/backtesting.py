"""Backtesting engine: simulate trading decisions over historical data."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.instruments import InstrumentSpec
from src.risk_management import RiskManager, Trade

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtester for a long-only XGBoost trading strategy.

    Parameters
    ----------
    initial_capital:
        Starting portfolio value in dollars.
    commission:
        Round-trip commission rate (e.g. 0.001 = 0.1 % per side).
    slippage:
        Assumed slippage rate applied to each fill price.
    risk_manager:
        A :class:`~src.risk_management.RiskManager` instance.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_manager: Optional[RiskManager] = None,
        instrument_spec: Optional[InstrumentSpec] = None,
        take_profit_points: float = 100.0,
        stop_loss_points: float = 20.0,
        same_bar_rule: str = "drop",
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = risk_manager or RiskManager()
        self.instrument_spec = instrument_spec or InstrumentSpec(symbol="UNKNOWN")
        self.take_profit_points = take_profit_points
        self.stop_loss_points = stop_loss_points
        self.same_bar_rule = same_bar_rule.lower()

        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate trading on *df* and return a portfolio equity DataFrame.

        Parameters
        ----------
        df:
            Signal-annotated DataFrame with columns including
            ``Open, High, Low, Close, signal, atr`` (atr optional).
            The ``signal`` column must contain 1 (BUY) or 0 (NO-BUY).

        Returns
        -------
        pd.DataFrame
            Row-per-day equity curve with columns
            ``portfolio_value, cash, position_value, returns, drawdown``.
        """
        self._reset()
        equity_records: List[Dict] = []

        for date, row in df.iterrows():
            # ---- check existing position for stop/take-profit exits ----
            if self.current_trade is not None:
                self._check_exit(date, row)

            # ---- open new position on BUY signal (no open position) ----
            if self.current_trade is None and row.get("signal", 0) == 1:
                can_trade = self.risk_manager.check_drawdown_limit(
                    self.cash + self._position_value(row["Close"]),
                    self.peak_value,
                )
                if can_trade:
                    self._open_trade(date, row)

            # ---- end-of-bar portfolio snapshot ----
            portfolio_val = self.cash + self._position_value(row["Close"])
            self.peak_value = max(self.peak_value, portfolio_val)
            drawdown = (self.peak_value - portfolio_val) / self.peak_value if self.peak_value else 0
            equity_records.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_val,
                    "cash": self.cash,
                    "position_value": self._position_value(row["Close"]),
                    "drawdown": drawdown,
                }
            )

        # Close any open position at last bar
        if self.current_trade is not None and not df.empty:
            last_row = df.iloc[-1]
            self._close_trade(df.index[-1], last_row["Close"], reason="end_of_data")

        result = pd.DataFrame(equity_records).set_index("date")
        result["returns"] = result["portfolio_value"].pct_change().fillna(0)
        return result

    def performance_metrics(self, equity: pd.DataFrame) -> Dict[str, float]:
        """Compute key performance metrics from an equity curve DataFrame."""
        portfolio_values = equity["portfolio_value"]
        returns = equity["returns"]

        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        trading_days = len(returns)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        daily_std = returns.std()
        sharpe = (returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        neg_returns = returns[returns < 0]
        sortino_denom = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0
        sortino = (returns.mean() * 252 / sortino_denom) if sortino_denom > 0 else 0

        max_drawdown = equity["drawdown"].max()

        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        win_trades = [t for t in self.trades if t.pnl > 0]
        total_trades = len(self.trades)
        win_rate = len(win_trades) / total_trades if total_trades else 0

        gross_profit = sum(t.pnl for t in win_trades)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
        avg_loss_trades = [t for t in self.trades if t.pnl < 0]
        avg_loss = np.mean([t.pnl for t in avg_loss_trades]) if avg_loss_trades else 0

        return {
            "total_return": round(total_return * 100, 2),
            "annual_return": round(annual_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "calmar_ratio": round(calmar, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "final_portfolio_value": round(portfolio_values.iloc[-1], 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self.cash = self.initial_capital
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.peak_value = self.initial_capital

    def _position_value(self, price: float) -> float:
        if self.current_trade is None:
            return 0.0
        return self.current_trade.quantity * price

    def _open_trade(self, date: pd.Timestamp, row: pd.Series) -> None:
        entry_price = row["Close"]
        fill_price = entry_price * (1 + self.slippage)
        atr = row.get("atr", None)
        quantity = self.risk_manager.calculate_position_size(
            self.cash, fill_price, atr=atr
        )
        if quantity <= 0:
            return

        cost = quantity * fill_price * (1 + self.commission)
        if cost > self.cash:
            quantity = self.cash / (fill_price * (1 + self.commission))
            cost = quantity * fill_price * (1 + self.commission)

        if quantity <= 0:
            return

        tp_price, sl_price = self._resolve_tp_sl_prices(entry_price, 1)
        self.cash -= cost
        self.current_trade = Trade(
            entry_date=date,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )
        logger.debug("OPEN  %s @ %.2f  qty=%.0f  SL=%.2f  TP=%.2f",
                     date, entry_price, quantity,
                     self.current_trade.stop_loss_price,
                     self.current_trade.take_profit_price)

    def _check_exit(self, date: pd.Timestamp, row: pd.Series) -> tuple[bool, str]:
        if self.current_trade is None:
            return False, ""

        tp_price, sl_price = self._resolve_tp_sl_prices(self.current_trade.entry_price, 1)
        low = row["Low"]
        high = row["High"]
        if low <= sl_price and high >= tp_price:
            if self.same_bar_rule == "tp_first":
                self._close_trade(date, tp_price, reason="take_profit")
                return True, "take_profit"
            if self.same_bar_rule == "sl_first":
                self._close_trade(date, sl_price, reason="stop_loss")
                return True, "stop_loss"
            return False, "same_bar"

        if high >= tp_price:
            self._close_trade(date, tp_price, reason="take_profit")
            return True, "take_profit"

        if low <= sl_price:
            self._close_trade(date, sl_price, reason="stop_loss")
            return True, "stop_loss"

        return False, ""

    def _close_trade(
        self, date: pd.Timestamp, price: float, reason: str = "signal"
    ) -> None:
        trade = self.current_trade
        if trade is None:
            return

        exit_price = price
        proceeds = self._calculate_proceeds(trade.quantity, trade.entry_price, exit_price)
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_reason = reason
        entry_cost = trade.quantity * trade.entry_price
        trade.pnl = proceeds - entry_cost
        trade.return_pct = trade.pnl / entry_cost if entry_cost else 0
        self.cash += proceeds
        self.trades.append(trade)
        self.current_trade = None
        logger.debug("CLOSE %s @ %.2f  pnl=%.2f  reason=%s",
                     date, exit_price, trade.pnl, reason)

    def _resolve_tp_sl_prices(self, entry_price: float, direction: int = 1) -> tuple[float, float]:
        point_size = self.instrument_spec.effective_point_size()
        tp_price = entry_price + self.take_profit_points * point_size
        sl_price = entry_price - self.stop_loss_points * point_size
        if direction < 0:
            tp_price = entry_price - self.take_profit_points * point_size
            sl_price = entry_price + self.stop_loss_points * point_size
        return tp_price, sl_price

    def _calculate_proceeds(self, quantity: float, entry_price: float, exit_price: float) -> float:
        return quantity * exit_price * (1 - self.commission)
