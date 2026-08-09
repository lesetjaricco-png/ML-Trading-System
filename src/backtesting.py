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
    """Event-driven backtester for a three-way XGBoost trading strategy.

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
        self.same_bar_rule = (same_bar_rule or "drop").lower()
        if self.same_bar_rule == "conservative_sl":
            self.same_bar_rule = "sl_first"

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
            signal = int(row.get("signal", 0))
            if signal != 0:
                self.diagnostics["signals_received"] += 1

            exited_this_bar = False

            # ---- check existing position for stop/take-profit exits ----
            if self.current_trade is not None:
                should_exit, reason = self._check_exit(date, row)
                if should_exit:
                    self.diagnostics[f"{reason}_exits"] += 1
                    exited_this_bar = True

            if self.open_position_count > 1:
                raise RuntimeError(f"Open position count invariant violated: {self.open_position_count}")

            # ---- open/flip a new position based on the signal (no open position) ----
            if self.current_trade is None and signal != 0 and not exited_this_bar:
                self.diagnostics["entries_attempted"] += 1
                can_trade = self.risk_manager.check_drawdown_limit(
                    self.cash + self._position_value(row["Close"]),
                    self.peak_value,
                )
                if can_trade:
                    opened = self._open_trade(date, row)
                    if opened:
                        self.diagnostics["entries_accepted"] += 1
                    else:
                        self.diagnostics["entries_rejected"] += 1
                else:
                    self.diagnostics["entries_rejected"] += 1
            elif self.current_trade is not None and signal != 0:
                self.diagnostics["entries_rejected_because_position_already_open"] += 1

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

        # Close any open position at last bar and update the last snapshot
        if self.current_trade is not None and not df.empty:
            last_row = df.iloc[-1]
            self._close_trade(df.index[-1], last_row["Close"], reason="end_of_data")
            self.diagnostics["end_of_data_exits"] += 1
            if equity_records:
                portfolio_val = self.cash + self._position_value(last_row["Close"])
                self.peak_value = max(self.peak_value, portfolio_val)
                drawdown = (self.peak_value - portfolio_val) / self.peak_value if self.peak_value else 0
                equity_records[-1].update(
                    {
                        "cash": self.cash,
                        "position_value": self._position_value(last_row["Close"]),
                        "portfolio_value": portfolio_val,
                        "drawdown": drawdown,
                    }
                )

        result = pd.DataFrame(equity_records).set_index("date")
        result["returns"] = result["portfolio_value"].pct_change().fillna(0)
        return result

    def performance_metrics(self, equity: pd.DataFrame) -> Dict[str, float]:
        """Compute key performance metrics from an equity curve DataFrame."""
        portfolio_values = equity["portfolio_value"]
        returns = equity["returns"]

        final_portfolio_value = self.calculate_independent_portfolio_value()
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
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
            "final_portfolio_value": round(final_portfolio_value, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def calculate_independent_portfolio_value(self, initial_capital: Optional[float] = None) -> float:
        """Return the portfolio value from the realized trade ledger alone."""
        base_capital = self.initial_capital if initial_capital is None else initial_capital
        return float(base_capital + sum(trade.net_pnl for trade in self.trades))

    def _reset(self) -> None:
        self.cash = self.initial_capital
        self.current_trade: Optional[Trade] = None
        self.open_position_count = 0
        self.max_simultaneous_positions = 0
        self.trades: List[Trade] = []
        self.peak_value = self.initial_capital
        self.diagnostics = {
            "signals_received": 0,
            "entries_attempted": 0,
            "entries_accepted": 0,
            "entries_rejected": 0,
            "entries_rejected_because_position_already_open": 0,
            "take_profit_exits": 0,
            "stop_loss_exits": 0,
            "end_of_data_exits": 0,
        }

    def _position_value(self, price: float) -> float:
        if self.current_trade is None:
            return 0.0
        point_size = self.instrument_spec.effective_point_size()
        tick_value = self.instrument_spec.tick_value
        if tick_value is None or tick_value <= 0:
            tick_value = self.instrument_spec.contract_size or 1.0
        return self.current_trade.quantity * float(price) / point_size * float(tick_value)

    def _open_trade(self, date: pd.Timestamp, row: pd.Series) -> bool:
        if self.current_trade is not None:
            return False

        entry_price = float(row["Close"])
        direction = int(row.get("signal", 0))
        if direction == 0 or not np.isfinite(entry_price):
            return False

        fill_price = self._apply_slippage(entry_price, direction, is_entry=True)
        atr = row.get("atr", None)
        quantity = self.risk_manager.calculate_position_size(
            self.cash,
            fill_price,
            atr=atr,
            instrument_spec=self.instrument_spec,
            stop_loss_points=self.stop_loss_points,
        )
        if not np.isfinite(quantity) or quantity <= 0:
            return False

        if self.risk_manager.max_position_size > 0 and quantity > self.risk_manager.max_position_size + 1e-12:
            return False

        point_size = self.instrument_spec.effective_point_size()
        tick_value = self.instrument_spec.tick_value
        if tick_value is None or tick_value <= 0:
            tick_value = self.instrument_spec.contract_size or 1.0

        entry_cost = abs(quantity) * fill_price / point_size * float(tick_value)
        entry_commission = entry_cost * self.commission
        entry_slippage = abs(quantity) * abs(fill_price - entry_price) / point_size * float(tick_value)
        entry_total_cost = entry_cost + entry_commission + entry_slippage
        if entry_total_cost > self.cash + 1e-12:
            return False

        tp_price, sl_price = self._resolve_tp_sl_prices(entry_price, direction)
        if direction > 0:
            self.cash -= entry_total_cost
        else:
            self.cash += entry_cost - entry_commission - entry_slippage

        self.current_trade = Trade(
            entry_date=date,
            entry_price=entry_price,
            quantity=quantity * direction,
            direction=direction,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            entry_fill_price=fill_price,
        )
        self.open_position_count = 1
        self.max_simultaneous_positions = max(self.max_simultaneous_positions, self.open_position_count)
        if self.open_position_count > 1:
            raise RuntimeError(f"Open position count invariant violated: {self.open_position_count}")
        logger.debug("OPEN  %s @ %.2f  qty=%.4f  SL=%.2f  TP=%.2f",
                     date, entry_price, quantity,
                     self.current_trade.stop_loss_price,
                     self.current_trade.take_profit_price)
        return True

    def _check_exit(self, date: pd.Timestamp, row: pd.Series) -> tuple[bool, str]:
        if self.current_trade is None:
            return False, ""

        tp_price, sl_price = self._resolve_tp_sl_prices(self.current_trade.entry_price, self.current_trade.direction)
        low = row["Low"]
        high = row["High"]
        if self.current_trade.direction > 0:
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price
        else:
            tp_hit = low <= tp_price
            sl_hit = high >= sl_price

        if tp_hit and sl_hit:
            if self.same_bar_rule in {"tp_first", "take_profit_first"}:
                self._close_trade(date, tp_price, reason="take_profit")
                return True, "take_profit"
            if self.same_bar_rule in {"sl_first", "conservative_sl", "stop_loss_first"}:
                self._close_trade(date, sl_price, reason="stop_loss")
                return True, "stop_loss"
            return False, "same_bar"

        if tp_hit:
            self._close_trade(date, tp_price, reason="take_profit")
            return True, "take_profit"

        if sl_hit:
            self._close_trade(date, sl_price, reason="stop_loss")
            return True, "stop_loss"

        return False, ""

    def _close_trade(
        self, date: pd.Timestamp, price: float, reason: str = "signal"
    ) -> None:
        trade = self.current_trade
        if trade is None:
            return

        exit_price = float(price)
        exit_fill_price = self._apply_slippage(exit_price, trade.direction, is_entry=False)
        point_size = self.instrument_spec.effective_point_size()
        tick_value = self.instrument_spec.tick_value
        if tick_value is None or tick_value <= 0:
            tick_value = self.instrument_spec.contract_size or 1.0
        entry_notional = abs(trade.quantity) * trade.entry_fill_price / point_size * float(tick_value)
        exit_notional = abs(trade.quantity) * exit_fill_price / point_size * float(tick_value)
        entry_commission = entry_notional * self.commission
        exit_commission = exit_notional * self.commission
        entry_slippage = abs(trade.quantity) * abs(trade.entry_fill_price - trade.entry_price) / point_size * float(tick_value)
        exit_slippage = abs(trade.quantity) * abs(exit_fill_price - exit_price) / point_size * float(tick_value)
        gross_pnl = self._calculate_trade_pnl(trade.quantity, trade.entry_fill_price, exit_fill_price)
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_fill_price = exit_fill_price
        trade.exit_reason = reason
        trade.gross_pnl = gross_pnl
        trade.commission = entry_commission + exit_commission
        trade.slippage = entry_slippage + exit_slippage
        trade.net_pnl = gross_pnl - trade.commission - trade.slippage
        if trade.direction > 0:
            self.cash += exit_notional - exit_commission - exit_slippage
        else:
            self.cash -= exit_notional + exit_commission + exit_slippage

        trade.pnl = trade.net_pnl
        trade.return_pct = trade.pnl / (abs(trade.quantity) * trade.entry_fill_price) if abs(trade.quantity) * trade.entry_fill_price else 0
        self.trades.append(trade)
        self.current_trade = None
        self.open_position_count = 0
        if self.open_position_count > 1:
            raise RuntimeError(f"Open position count invariant violated: {self.open_position_count}")
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

    def _apply_slippage(self, price: float, direction: int, is_entry: bool) -> float:
        if not np.isfinite(price):
            return price
        if direction > 0:
            return price * (1 + self.slippage) if is_entry else price * (1 - self.slippage)
        return price * (1 - self.slippage) if is_entry else price * (1 + self.slippage)

    def _calculate_proceeds(self, quantity: float, entry_price: float, exit_price: float) -> float:
        return quantity * exit_price * (1 - self.commission)

    def _calculate_trade_pnl(self, quantity: float, entry_price: float, exit_price: float) -> float:
        point_size = self.instrument_spec.effective_point_size()
        tick_value = self.instrument_spec.tick_value
        if tick_value is None or tick_value <= 0:
            tick_value = self.instrument_spec.contract_size or 1.0
        price_move = exit_price - entry_price
        return (quantity * price_move / point_size) * tick_value
