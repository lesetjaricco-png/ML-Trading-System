from __future__ import annotations

import sys
import types

import pandas as pd

from src.data_ingestion import DataIngestion


def test_fetch_uses_mt5_when_configured(tmp_path, monkeypatch):
    fake_mt5 = types.ModuleType("MetaTrader5")
    fake_mt5.TIMEFRAME_M15 = "M15"
    fake_mt5.initialize = lambda *args, **kwargs: True
    fake_mt5.shutdown = lambda: None
    fake_mt5.symbol_select = lambda *args, **kwargs: True
    fake_mt5.copy_rates_from = lambda *args, **kwargs: [
        {
            "time": 1704067200,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 1200,
        }
    ]
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)

    ingestion = DataIngestion(data_dir=str(tmp_path))
    df = ingestion.fetch(
        ticker="US30",
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="15m",
        source="mt5",
        use_cache=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.iloc[0]["Close"] == 100.5
    assert df.index.name == "Date"
