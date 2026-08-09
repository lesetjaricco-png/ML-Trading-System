from __future__ import annotations

import pandas as pd
import pytest

import main
from main import build_diagnostic_report, select_backtest_subset
from src.data_ingestion import IngestionError


def test_select_backtest_subset_uses_holdout_period():
    df = pd.DataFrame({"Close": range(10)}, index=pd.date_range("2024-01-01", periods=10, freq="15min"))

    subset = select_backtest_subset(df, test_size=0.2, validation_size=0.1)
    expected = df.iloc[8:].copy()

    assert len(subset) == 2
    assert subset.index.equals(expected.index)
    assert subset.index[0] == df.index[8]
    assert subset.index[-1] == df.index[9]


def test_build_diagnostic_report_contains_expected_sections():
    df_raw = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [1, 2],
            "Close": [2, 3],
            "Volume": [10, 20],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="15min"),
    )
    df_features = df_raw.assign(target=[1, 0], signal=[1, 0], signal_proba=[0.9, 0.2])
    equity = pd.DataFrame(
        {
            "portfolio_value": [100000.0, 101000.0],
            "cash": [100000.0, 100000.0],
            "position_value": [0.0, 1000.0],
            "drawdown": [0.0, 0.0],
        },
        index=df_raw.index,
    )
    equity["returns"] = equity["portfolio_value"].pct_change().fillna(0)

    report = build_diagnostic_report(
        df_raw=df_raw,
        df_features=df_features,
        split_summary={"train_rows": 1, "val_rows": 0, "test_rows": 1},
        df_signals=df_features,
        equity=equity,
        perf={"total_trades": 1, "final_portfolio_value": 101000.0},
    )

    assert report["data"]["rows"] == 2
    assert report["features"]["feature_rows"] == 2
    assert report["signals"]["buy_signals"] == 1
    assert report["backtest"]["total_trades"] == 1


def test_build_diagnostic_report_counts_three_way_signals():
    df_raw = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [1, 2, 3],
            "Close": [2, 3, 4],
            "Volume": [10, 20, 30],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="15min"),
    )
    df_signals = pd.DataFrame({"signal": [1, -1, 0]})

    report = build_diagnostic_report(
        df_raw=df_raw,
        df_features=df_raw.assign(target=[1, 0, 1]),
        split_summary={"train_rows": 1, "val_rows": 1, "test_rows": 1},
        df_signals=df_signals,
        equity=pd.DataFrame({"portfolio_value": [100000.0, 100000.0, 100000.0]}),
        perf={"total_trades": 0, "final_portfolio_value": 100000.0},
    )

    assert report["signals"]["buy_signals"] == 1
    assert report["signals"]["sell_signals"] == 1
    assert report["signals"]["no_trade_signals"] == 1


def test_pipeline_stops_when_ingestion_fails(monkeypatch):
    class FakeIngestion:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, *args, **kwargs):
            raise IngestionError("MT5 unavailable")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Feature engineering should not run after ingestion failure")

    monkeypatch.setattr(main, "DataIngestion", FakeIngestion)
    monkeypatch.setattr(main.FeatureEngineer, "transform", fail_if_called)

    cfg = {
        "data": {
            "ticker": "US30",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "interval": "15m",
            "source": "mt5",
        },
        "features": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2,
            "sma_periods": [10],
            "ema_periods": [12],
            "atr_period": 14,
            "volume_sma_period": 20,
        },
        "target": {
            "take_profit_points": 100,
            "stop_loss_points": 20,
            "max_bars": 40,
            "same_bar_rule": "drop",
            "unresolved_policy": "drop",
        },
        "instruments": {},
        "experiment": {"target_mode": "v0.3_forward_atr", "signal_mode": "v0.3_forward_atr"},
        "model": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "eval_metric": "logloss",
            "early_stopping_rounds": 5,
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        "signal": {"prediction_threshold": 0.55, "buy_threshold": 0.7, "sell_threshold": 0.7},
        "risk": {
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "max_drawdown": 0.15,
            "risk_per_trade": 0.01,
        },
        "backtest": {"initial_capital": 100000, "commission": 0.001, "slippage": 0.0005},
    }

    with pytest.raises(IngestionError, match="MT5 unavailable"):
        main.run_pipeline(cfg, use_cache=False)
