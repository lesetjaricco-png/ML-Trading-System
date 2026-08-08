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
        dates = pd.date_range("2024-01-01", periods=6, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104, 105],
                "High": [100, 105, 102, 103, 104, 105],
                "Low": [100, 99, 95, 97, 102, 103],
                "Close": [100, 102, 98, 99, 103, 104],
                "Volume": [1000, 1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        fe = FeatureEngineer(
            timeframe="15m",
            take_profit_points=5,
            stop_loss_points=2,
            max_bars=1,
            same_bar_rule="drop",
            unresolved_policy="drop",
        )
        result = fe.transform(df)
        assert "target" in result.columns
        assert set(result["target"].unique()).issubset({0, 1})
        assert 0 in result["target"].unique()
        assert 1 in result["target"].unique()

    def test_instrument_specific_target_config_is_used(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="15min")
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [100, 104],
                "Low": [100, 100],
                "Close": [100, 101],
                "Volume": [1000, 1000],
            },
            index=dates,
        )
        fe = FeatureEngineer(
            timeframe="15m",
            take_profit_points=100,
            stop_loss_points=20,
            max_bars=1,
            same_bar_rule="drop",
            unresolved_policy="drop",
            instrument_config={"US30": {"take_profit_points": 4, "stop_loss_points": 1}},
        )
        result = fe.transform(df, instrument_name="US30")
        assert "target" in result.columns
        assert len(result) > 0
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

    def test_feature_columns_present(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        for col in fe.feature_columns:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_rsi_range(self, sample_ohlcv):
        fe = FeatureEngineer()
        result = fe.transform(sample_ohlcv)
        assert result["rsi"].between(0, 100).all()

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
        assert (result["volume_ratio"] > 0).all()

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
