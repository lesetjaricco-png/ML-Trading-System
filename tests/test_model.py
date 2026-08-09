"""Tests for the XGBoost model module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import src.model as model_module
from src.feature_engineering import FeatureEngineer
from src.model import XGBoostModel


@pytest.fixture
def feature_df(sample_ohlcv):
    fe = FeatureEngineer(sma_periods=[10, 20, 50])
    return fe.transform(sample_ohlcv)


class TestXGBoostModel:
    def test_fit_training_data_fits_scaler_only_on_supplied_rows(self, tmp_path):
        training_df = pd.DataFrame(
            {
                "feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "target": [0, 1, 2, 0, 1, 2],
            }
        )
        future_rows = pd.DataFrame(
            {
                "feature": [1000.0, 2000.0],
                "target": [0, 1],
            }
        )
        model = XGBoostModel(n_estimators=2, max_depth=1, models_dir=str(tmp_path))

        model.fit_training_data(training_df, ["feature"])

        np.testing.assert_allclose(model.scaler.mean_, [2.5])
        assert model.scaler.mean_[0] != pd.concat([training_df, future_rows])["feature"].mean()
        assert model.is_fitted

    def test_cross_validate_fits_scaler_on_each_training_fold_only(
        self, monkeypatch, tmp_path
    ):
        feature_values = np.array(
            [0, 1, 2, 3, 4, 5, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            dtype=float,
        )
        df = pd.DataFrame(
            {
                "chronological_feature": feature_values,
                "target": np.tile([0, 1], len(feature_values) // 2),
            }
        )
        scaler_fit_inputs = []
        scaler_means = []

        class RecordingScaler:
            def __init__(self):
                self.scaler = StandardScaler()

            def fit(self, values):
                scaler_fit_inputs.append(values.copy())
                self.scaler.fit(values)
                scaler_means.append(self.scaler.mean_.copy())
                return self

            def transform(self, values):
                return self.scaler.transform(values)

        class StubClassifier:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, verbose=False):
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                return np.tile([0.5, 0.5], (len(X), 1))

        monkeypatch.setattr(model_module, "StandardScaler", RecordingScaler)
        monkeypatch.setattr(model_module, "XGBClassifier", StubClassifier)

        model = XGBoostModel(models_dir=str(tmp_path))
        results = model.cross_validate(df, ["chronological_feature"], n_splits=3)
        expected_splits = list(TimeSeriesSplit(n_splits=3).split(feature_values))

        assert len(results) == len(expected_splits)
        assert len(scaler_fit_inputs) == len(expected_splits)
        for fit_values, scaler_mean, (train_idx, validation_idx) in zip(
            scaler_fit_inputs, scaler_means, expected_splits
        ):
            expected_train = feature_values[train_idx, np.newaxis]
            np.testing.assert_array_equal(fit_values, expected_train)
            np.testing.assert_allclose(scaler_mean, expected_train.mean(axis=0))
            assert train_idx[-1] < validation_idx[0]
            assert fit_values[-1, 0] < feature_values[validation_idx[0]]

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

    def test_predict_signal_supports_three_way_decisions(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.train(feature_df, fe.feature_columns)
        X = feature_df[fe.feature_columns].values
        signals = model.predict_signal(X)
        assert set(signals).issubset({-1, 0, 1})

    def test_generate_signals_adds_columns(self, feature_df):
        fe = FeatureEngineer(sma_periods=[10, 20, 50])
        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.train(feature_df, fe.feature_columns)
        result = model.generate_signals(feature_df)
        assert "signal" in result.columns
        assert "signal_proba" in result.columns

    def test_generate_signals_supports_v03_three_way_probabilities(self, sample_ohlcv):
        fe = FeatureEngineer(
            sma_periods=[3],
            target_mode="v0.3_forward_atr",
            forward_horizon=2,
            atr_threshold_multiplier=0.0,
        )
        feature_df = fe.transform(sample_ohlcv, instrument_name="US30")
        model = XGBoostModel(n_estimators=50, max_depth=3, signal_mode="v0.3_forward_atr")
        model.train(feature_df, fe.feature_columns)
        result = model.generate_signals(feature_df)
        assert "buy_proba" in result.columns
        assert "sell_proba" in result.columns
        assert "no_trade_proba" in result.columns

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
