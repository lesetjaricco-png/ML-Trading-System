from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InstrumentSpec:
    """Minimal instrument metadata for TP/SL and P&L conversion."""

    symbol: str
    point_size: float = 0.01
    tick_size: float | None = None
    tick_value: float | None = None
    contract_size: float | None = None
    volume_min: float | None = None
    volume_step: float | None = None
    digits: int = 5

    def effective_point_size(self) -> float:
        return self.point_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "point_size": self.point_size,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "contract_size": self.contract_size,
            "volume_min": self.volume_min,
            "volume_step": self.volume_step,
            "digits": self.digits,
        }


def resolve_instrument_spec(symbol: str, *, fallback_point_size: float = 0.01) -> InstrumentSpec:
    """Resolve a simple instrument specification from MetaTrader5 metadata when available."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return InstrumentSpec(symbol=symbol, point_size=fallback_point_size)

    try:
        info = mt5.symbol_info(symbol)
    except Exception:  # pragma: no cover - runtime environment dependent
        return InstrumentSpec(symbol=symbol, point_size=fallback_point_size)

    if info is None:
        return InstrumentSpec(symbol=symbol, point_size=fallback_point_size)

    return InstrumentSpec(
        symbol=getattr(info, "name", symbol) or symbol,
        point_size=float(getattr(info, "point", fallback_point_size) or fallback_point_size),
        tick_size=float(getattr(info, "trade_tick_size", None)) if getattr(info, "trade_tick_size", None) is not None else None,
        tick_value=float(getattr(info, "trade_tick_value", None)) if getattr(info, "trade_tick_value", None) is not None else None,
        contract_size=float(getattr(info, "trade_contract_size", None)) if getattr(info, "trade_contract_size", None) is not None else None,
        volume_min=float(getattr(info, "volume_min", None)) if getattr(info, "volume_min", None) is not None else None,
        volume_step=float(getattr(info, "volume_step", None)) if getattr(info, "volume_step", None) is not None else None,
        digits=int(getattr(info, "digits", 5) or 5),
    )
