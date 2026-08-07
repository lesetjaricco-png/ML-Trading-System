"""Tests for the XGBoost model module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer
from src.model import XGBoostModel


@pytest.fixture
def feature_df(sample_ohlcv):
    fe = FeatureEngineer(sma_periods=[10, 20, 50])
    return fe.transform(sample_ohlcv)


class TestXGBoostModel:
    def test_train_returns_metrics(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        metrics = model.train(
            feature_df,
            feature_columns=fe.feature_columns,
            test_size=0.2,
            validation_size=0.1,
        )
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_predict_proba_range(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.train(feature_df, fe.feature_columns)
        X = feature_df[fe.feature_columns].values
        proba = model.predict_proba(X)
        assert proba.shape == (len(feature_df),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_signal_binary(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.train(feature_df, fe.feature_columns)
        X = feature_df[fe.feature_columns].values
        signals = model.predict_signal(X)
        assert set(signals).issubset({0, 1})

    def test_generate_signals_adds_columns(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.train(feature_df, fe.feature_columns)
        result = model.generate_signals(feature_df)
        assert "signal" in result.columns
        assert "signal_proba" in result.columns

    def test_feature_importance_shape(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.train(feature_df, fe.feature_columns)
        fi = model.feature_importance()
        assert len(fi) == len(fe.feature_columns)
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_predict_before_train_raises(self, feature_df):
        model = XGBoostModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(feature_df.values[:5])

    def test_save_load_roundtrip(self, feature_df, tmp_path):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3, models_dir=str(tmp_path))
        model.train(feature_df, fe.feature_columns)
        model.save("test_model")

        model2 = XGBoostModel(models_dir=str(tmp_path))
        model2.load("test_model")
        X = feature_df[fe.feature_columns].values
        p1 = model.predict_proba(X)
        p2 = model2.predict_proba(X)
        np.testing.assert_array_almost_equal(p1, p2)
