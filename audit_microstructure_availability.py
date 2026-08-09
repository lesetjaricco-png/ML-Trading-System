"""Read-only audit of reproducible US30 M1 and tick history in MT5."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import MetaTrader5 as mt5


SYMBOL = "US30"
PROBE_STARTS = (
    datetime(2022, 5, 12, tzinfo=timezone.utc),
    datetime(2024, 1, 15, tzinfo=timezone.utc),
    datetime(2025, 10, 2, tzinfo=timezone.utc),
)
SPEC_FIELDS = (
    "name",
    "description",
    "path",
    "currency_base",
    "currency_profit",
    "digits",
    "point",
    "trade_tick_size",
    "trade_tick_value",
    "trade_contract_size",
    "spread",
    "spread_float",
    "ticks_bookdepth",
)


def _array_summary(values: Any) -> dict[str, Any]:
    return {
        "rows": None if values is None else len(values),
        "fields": None if values is None else list(values.dtype.names or ()),
    }


def audit() -> dict[str, Any]:
    result: dict[str, Any] = {
        "symbol": SYMBOL,
        "initialize_ok": bool(mt5.initialize()),
        "symbol_select_ok": False,
        "contract_specification": None,
        "probes": [],
        "last_error": None,
    }
    if not result["initialize_ok"]:
        result["last_error"] = list(mt5.last_error())
        return result

    try:
        result["symbol_select_ok"] = bool(mt5.symbol_select(SYMBOL, True))
        info = mt5.symbol_info(SYMBOL)
        if info is not None:
            result["contract_specification"] = {
                field: getattr(info, field, None) for field in SPEC_FIELDS
            }
        for start in PROBE_STARTS:
            rates = mt5.copy_rates_range(
                SYMBOL, mt5.TIMEFRAME_M1, start, start + timedelta(days=1)
            )
            ticks = mt5.copy_ticks_range(
                SYMBOL, start, start + timedelta(hours=1), mt5.COPY_TICKS_ALL
            )
            result["probes"].append(
                {
                    "start_utc": start.isoformat(),
                    "m1": _array_summary(rates),
                    "ticks": _array_summary(ticks),
                }
            )
        result["last_error"] = list(mt5.last_error())
        return result
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))