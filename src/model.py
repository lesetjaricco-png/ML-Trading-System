"""XGBoost model: training, evaluation, and signal generation."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class XGBoostModel:
    """Wrapper around XGBClassifier for binary trading signal prediction.

    Parameters
    ----------
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight, gamma, reg_alpha, reg_lambda:
        Standard XGBoost hyper-parameters.
    random_state:
        Seed for reproducibility.
    eval_metric:
        XGBoost evaluation metric (``"logloss"``).
    early_stopping_rounds:
        Stop training if the validation metric does not improve.
    prediction_threshold:
        Probability threshold for a BUY signal (default 0.55).
    models_dir:
        Directory to save/load model artefacts.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        eval_metric: str = "logloss",
        early_stopping_rounds: int = 50,
        prediction_threshold: float = 0.55,
        models_dir: str = "models",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.prediction_threshold = prediction_threshold
        self.models_dir = models_dir

        self.model: Optional[XGBClassifier] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

        os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> Dict[str, float]:
        """Train the XGBoost classifier using a time-series split.

        Parameters
        ----------
        df:
            Feature-engineered DataFrame that includes a ``target`` column.
        feature_columns:
            Names of the columns to use as model features.
        test_size:
            Fraction of data held out for final evaluation.
        validation_size:
            Fraction of training data used for early-stopping validation.

        Returns
        -------
        dict
            Evaluation metrics on the held-out test set.
        """
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        y = df["target"].values

        # Time-series split (no shuffling)
        n = len(X)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]

        logger.info(
            "Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test)
        )

        # Scale features
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=-1,
        )

        self.model.fit(
            X_train_s,
            y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )

        self.is_fitted = True
        metrics = self._evaluate(X_test_s, y_test)
        logger.info("Test metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Prediction / signal generation
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-1 probability for each sample."""
        self._check_fitted()
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)[:, 1]

    def predict_signal(self, X: np.ndarray) -> np.ndarray:
        """Return binary BUY (1) / NO-BUY (0) signals."""
        proba = self.predict_proba(X)
        return (proba >= self.prediction_threshold).astype(int)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``signal`` and ``signal_proba`` columns to *df*.

        Parameters
        ----------
        df:
            Feature-engineered DataFrame (must contain ``feature_columns``).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with two extra columns.
        """
        self._check_fitted()
        df = df.copy()
        X = df[self.feature_columns].values
        df["signal_proba"] = self.predict_proba(X)
        df["signal"] = (df["signal_proba"] >= self.prediction_threshold).astype(int)
        return df

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self, df: pd.DataFrame, feature_columns: List[str], n_splits: int = 5
    ) -> List[Dict[str, float]]:
        """Walk-forward cross-validation using ``TimeSeriesSplit``."""
        self.feature_columns = feature_columns
        X = self.scaler.fit_transform(df[feature_columns].values)
        y = df["target"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx], verbose=False)
            metrics = self._evaluate(X[test_idx], y[test_idx], model=model)
            metrics["fold"] = fold
            results.append(metrics)
            logger.info("Fold %d metrics: %s", fold, metrics)
        return results

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame with feature importances sorted descending."""
        self._check_fitted()
        importance = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "xgb_trading_model") -> str:
        """Save model and scaler to disk. Returns the saved file path."""
        self._check_fitted()
        path = os.path.join(self.models_dir, f"{name}.joblib")
        joblib.dump({"model": self.model, "scaler": self.scaler, "features": self.feature_columns}, path)
        logger.info("Model saved to %s", path)
        return path

    def load(self, name: str = "xgb_trading_model") -> None:
        """Load a previously saved model from disk."""
        path = os.path.join(self.models_dir, f"{name}.joblib")
        artefacts = joblib.load(path)
        self.model = artefacts["model"]
        self.scaler = artefacts["scaler"]
        self.feature_columns = artefacts["features"]
        self.is_fitted = True
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[XGBClassifier] = None,
    ) -> Dict[str, float]:
        m = model or self.model
        y_pred = m.predict(X)
        y_proba = m.predict_proba(X)[:, 1]
        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
        }

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
