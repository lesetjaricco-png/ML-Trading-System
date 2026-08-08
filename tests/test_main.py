from __future__ import annotations

import pandas as pd

from main import build_diagnostic_report, select_backtest_subset


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
