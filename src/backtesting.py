"""Backtesting engine: simulate trading decisions over historical data."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = risk_manager or RiskManager()

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
        entry_price = row["Close"] * (1 + self.slippage)
        atr = row.get("atr", None)
        quantity = self.risk_manager.calculate_position_size(
            self.cash, entry_price, atr=atr
        )
        if quantity <= 0:
            return

        cost = quantity * entry_price * (1 + self.commission)
        if cost > self.cash:
            quantity = np.floor(self.cash / (entry_price * (1 + self.commission)))
            cost = quantity * entry_price * (1 + self.commission)

        if quantity <= 0:
            return

        self.cash -= cost
        self.current_trade = Trade(
            entry_date=date,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_price=self.risk_manager.calculate_stop_loss(entry_price, atr=atr),
            take_profit_price=self.risk_manager.calculate_take_profit(entry_price, atr=atr),
        )
        logger.debug("OPEN  %s @ %.2f  qty=%.0f  SL=%.2f  TP=%.2f",
                     date, entry_price, quantity,
                     self.current_trade.stop_loss_price,
                     self.current_trade.take_profit_price)

    def _check_exit(self, date: pd.Timestamp, row: pd.Series) -> None:
        should_exit, reason = self.risk_manager.should_exit(
            self.current_trade, row["Low"]
        )
        if not should_exit:
            should_exit, reason = self.risk_manager.should_exit(
                self.current_trade, row["High"]
            )
        if should_exit:
            exit_price = (
                self.current_trade.stop_loss_price
                if reason == "stop_loss"
                else self.current_trade.take_profit_price
            )
            self._close_trade(date, exit_price, reason=reason)

    def _close_trade(
        self, date: pd.Timestamp, price: float, reason: str = "signal"
    ) -> None:
        trade = self.current_trade
        exit_price = price * (1 - self.slippage)
        proceeds = trade.quantity * exit_price * (1 - self.commission)
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
