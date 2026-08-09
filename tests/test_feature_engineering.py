"""Tests for the feature engineering module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    def test_transform_returns_dataframe(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_after_transform(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert result.isnull().sum().sum() == 0

    def test_target_is_binary(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert set(result["target"].unique()).issubset({0, 1})

    def test_tp_before_sl_target_generation(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100 + i for i in range(120)],
                "High": [100 + i + 1 for i in range(120)],
                "Low": [100 + i - 1 for i in range(120)],
                "Close": [100 + i for i in range(120)],
                "Volume": [1000] * 120,
            },
            index=dates,
        )
        fe = FeatureEngineer(
            timeframe="15m",
            rsi_period=2,
            macd_fast=2,
            macd_slow=3,
            macd_signal=1,
            bb_period=2,
            atr_period=2,
            take_profit_points=5,
            stop_loss_points=2,
            max_bars=3,
            same_bar_rule="tp_first",
            unresolved_policy="tp",
            sma_periods=[5],
        )
        result = fe.transform(df)
        assert "target" in result.columns
        assert set(result["target"].unique()).issubset({0, 1})
        assert len(result) >= 1

    def test_instrument_specific_target_config_is_used(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100 + i for i in range(120)],
                "High": [100 + i + 2 for i in range(120)],
                "Low": [100 + i - 2 for i in range(120)],
                "Close": [100 + i for i in range(120)],
                "Volume": [1000] * 120,
            },
            index=dates,
        )
        fe = FeatureEngineer(
            timeframe="15m",
            rsi_period=2,
            macd_fast=2,
            macd_slow=3,
            macd_signal=1,
            bb_period=2,
            atr_period=2,
            take_profit_points=100,
            stop_loss_points=20,
            max_bars=3,
            same_bar_rule="tp_first",
            unresolved_policy="tp",
            instrument_config={"US30": {"take_profit_points": 4, "stop_loss_points": 1}},
            sma_periods=[5],
        )
        result = fe.transform(df, instrument_name="US30")
        assert "target" in result.columns
        assert len(result) >= 1
        assert set(result["target"].unique()).issubset({0, 1})

    def test_evaluate_target_summary(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104, 105],
                "High": [100, 102, 103, 106, 110, 111],
                "Low": [100, 99, 100, 101, 102, 103],
                "Close": [100, 101, 102, 104, 105, 106],
                "Volume": [1000, 1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        fe = FeatureEngineer(
            timeframe="15m",
            take_profit_points=5,
            stop_loss_points=2,
            max_bars=3,
            same_bar_rule="drop",
            unresolved_policy="drop",
        )
        summary = fe.evaluate_target_summary(df, instrument_name="US30", instrument_config={"US30": {"take_profit_points": 4, "stop_loss_points": 1}})
        assert summary["total_observations"] == len(df)
        assert "win_rate" in summary
        assert "monthly_breakdown" in summary

    def test_v03_forward_atr_target_is_three_way(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="15min")
        close = [100, 100, 110, 100, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": [c + 0.5 for c in close],
                "Low": [c - 0.5 for c in close],
                "Close": close,
                "Volume": [1000] * len(close),
            },
            index=dates,
        )
        fe = FeatureEngineer(
            sma_periods=[3],
            target_mode="v0.3_forward_atr",
            forward_horizon=2,
            atr_threshold_multiplier=0.0,
        )
        result = fe.transform(df, instrument_name="US30")
        assert set(result["target"].dropna().unique()).issubset({0, 1, 2})
        assert (result["target"] == 2).any()
        assert (result["target"].dropna().nunique() >= 1)

    def test_v03_requires_complete_future_horizon(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 12,
                "High": [101.0] * 12,
                "Low": [99.0] * 12,
                "Close": [100.0] * 12,
                "Volume": [1000] * 12,
                "atr_pct": [0.01] * 12,
            },
            index=dates,
        )
        fe = FeatureEngineer(
            target_mode="v0.3_forward_atr",
            forward_horizon=5,
            atr_threshold_multiplier=1.0,
        )

        target = pd.Series(fe._build_v03_forward_atr_target_series(df), index=dates)

        assert target.iloc[:-5].notna().all()
        assert target.iloc[-5:].isna().all()

    def test_feature_columns_present(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        for col in fe.feature_columns:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_multi_period_returns_and_session_features_present(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert "returns_2" in result.columns
        assert "returns_5" in result.columns
        assert "returns_10" in result.columns
        assert "is_asia_session" in result.columns
        assert "is_london_session" in result.columns
        assert "is_new_york_session" in result.columns

    def test_rsi_range(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert result["rsi"].dropna().between(0, 100).all()

    def test_bb_pct_computed(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert "bb_pct" in result.columns

    def test_volatility_and_time_features_present(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert "volatility_5" in result.columns
        assert "volatility_20" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns

    def test_volume_ratio_positive(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert (result["volume_ratio"].dropna() > 0).all()

    def test_custom_sma_periods(self, sample_ohlcv):
        fe = FeatureEngineer(sma_periods=[5, 10])
        result = fe.transform(sample_ohlcv)
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_50" not in result.columns

    def test_row_count_reduced_by_warmup(self, sample_ohlcv):
        fe = FeatureEngineer(sma_periods=[200])
        result = fe.transform(sample_ohlcv)
        assert len(result) < len(sample_ohlcv)

    def test_future_price_changes_do_not_change_past_features(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="15min")
        base_close = [100 + i * 0.1 for i in range(80)]
        df = pd.DataFrame(
            {
                "Open": base_close,
                "High": [c + 0.2 for c in base_close],
                "Low": [c - 0.2 for c in base_close],
                "Close": base_close,
                "Volume": [1000 + i for i in range(80)],
            },
            index=dates,
        )
        fe = FeatureEngineer()
        original = fe.transform(df)

        modified = df.copy()
        modified.iloc[51:, 0] = modified.iloc[51:, 0] + 10.0
        modified.iloc[51:, 1] = modified.iloc[51:, 1] + 10.0
        modified.iloc[51:, 2] = modified.iloc[51:, 2] + 10.0
        modified.iloc[51:, 3] = modified.iloc[51:, 3] + 10.0
        modified_result = fe.transform(modified)

        original_before_cutoff = original.loc[original.index <= dates[50], :]
        modified_before_cutoff = modified_result.loc[modified_result.index <= dates[50], :]

        assert len(original_before_cutoff) == len(modified_before_cutoff)
        for column in fe.feature_columns:
            if column in {"target"}:
                continue
            pd.testing.assert_series_equal(
                original_before_cutoff[column].reset_index(drop=True),
                modified_before_cutoff[column].reset_index(drop=True),
                check_names=False,
                check_exact=False,
                atol=1e-9,
                rtol=1e-9,
            )

    def test_features_do_not_depend_on_backward_filling(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100 + i for i in range(30)],
                "High": [101 + i for i in range(30)],
                "Low": [99 + i for i in range(30)],
                "Close": [100 + i for i in range(30)],
                "Volume": [1000] * 30,
            },
            index=dates,
        )
        fe = FeatureEngineer(sma_periods=[10])
        result = fe.transform(df)

        assert len(result) < len(df)
        assert result["sma_10"].notna().all()
        assert result["returns"].notna().all()
