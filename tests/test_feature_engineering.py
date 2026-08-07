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
