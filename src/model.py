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
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.signal_decision import SignalDecisionEngine

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
        buy_threshold: float = 0.70,
        sell_threshold: float = 0.70,
        signal_mode: str = "v0.1_binary",
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
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.signal_mode = signal_mode.lower()
        self.decision_engine = SignalDecisionEngine(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        self.models_dir = models_dir

        self.model: Optional[XGBClassifier] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

        os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_training_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> None:
        """Fit the scaler and classifier using only the supplied training rows."""
        self.feature_columns = feature_columns
        invalid_feature_columns = [
            column
            for column in feature_columns
            if column in {"target"}
            or column.startswith("future_")
            or column.endswith("_outcome")
        ]
        if invalid_feature_columns:
            raise AssertionError(
                "Target or future-outcome columns cannot be used as model features: "
                f"{invalid_feature_columns}"
            )

        X_train = df[feature_columns].values
        y_train = df["target"].values
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        is_multiclass = len(np.unique(y_train)) > 2
        if is_multiclass:
            model_eval_metric = "mlogloss"
            model_objective = "multi:softprob"
            model_params = {"num_class": len(np.unique(y_train))}
        else:
            model_eval_metric = self.eval_metric
            model_objective = "binary:logistic"
            model_params = {}

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
            eval_metric=model_eval_metric,
            objective=model_objective,
            n_jobs=-1,
            **model_params,
        )
        self.model.fit(X_train_scaled, y_train, verbose=False)
        self.is_fitted = True

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
        invalid_feature_columns = [
            column for column in feature_columns if column in {"target"} or column.startswith("future_") or column.endswith("_outcome")
        ]
        if invalid_feature_columns:
            raise AssertionError(f"Target or future-outcome columns cannot be used as model features: {invalid_feature_columns}")
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

        is_multiclass = len(np.unique(y_train)) > 2
        if is_multiclass:
            model_eval_metric = "mlogloss"
            model_objective = "multi:softprob"
            model_params = {"num_class": len(np.unique(y_train))}
        else:
            model_eval_metric = self.eval_metric
            model_objective = "binary:logistic"
            model_params = {}

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
            eval_metric=model_eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            objective=model_objective,
            n_jobs=-1,
            **model_params,
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
        """Return class probabilities for the trained target scheme."""
        self._check_fitted()
        X_s = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_s)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities

    def predict_directional_probabilities(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return BUY and SELL probabilities for V0.2-style directional experiments."""
        self._check_fitted()
        X_s = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_s)
        if probabilities.ndim == 1:
            return probabilities, 1 - probabilities
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1], probabilities[:, 0]

        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            buy_index = None
            sell_index = None
            for idx, label in enumerate(classes):
                if label == 1:
                    buy_index = idx
                elif label == 0:
                    sell_index = idx
            if buy_index is not None and sell_index is not None:
                return probabilities[:, buy_index], probabilities[:, sell_index]

        if probabilities.shape[1] >= 3:
            return probabilities[:, 1], probabilities[:, 0]
        return probabilities[:, 0], probabilities[:, 0]

    def predict_signal(self, X: np.ndarray) -> np.ndarray:
        """Return a binary signal for V0.1 or a three-way signal for V0.2."""
        if self.signal_mode == "v0.2_directional":
            buy_prob, sell_prob = self.predict_directional_probabilities(X)
            return np.array([
                self.decision_engine.decide(float(buy), float(sell)) for buy, sell in zip(buy_prob, sell_prob)
            ], dtype=int)

        if self.signal_mode == "v0.3_forward_atr":
            probabilities = self.predict_proba(X)
            if probabilities.ndim == 1:
                return (probabilities >= self.prediction_threshold).astype(int)
            prob_array = np.asarray(probabilities)
            buy_prob = prob_array[:, 1] if prob_array.shape[1] > 1 else prob_array[:, 0]
            sell_prob = prob_array[:, 0] if prob_array.shape[1] > 1 else prob_array[:, 0]
            return np.array([
                self.decision_engine.decide(float(buy), float(sell)) for buy, sell in zip(buy_prob, sell_prob)
            ], dtype=int)

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
        if self.signal_mode == "v0.2_directional":
            buy_prob, sell_prob = self.predict_directional_probabilities(X)
            df["buy_proba"] = buy_prob
            df["sell_proba"] = sell_prob
            df["signal"] = self.predict_signal(X)
            df["signal_proba"] = np.where(df["signal"] == 1, buy_prob, np.where(df["signal"] == -1, sell_prob, 0.0))
        elif self.signal_mode == "v0.3_forward_atr":
            probabilities = np.asarray(self.predict_proba(X))
            if probabilities.ndim == 1:
                df["buy_proba"] = probabilities
                df["sell_proba"] = 1 - probabilities
                df["no_trade_proba"] = 0.0
            else:
                buy_prob = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                sell_prob = probabilities[:, 0] if probabilities.shape[1] > 1 else probabilities[:, 0]
                no_trade_prob = probabilities[:, 2] if probabilities.shape[1] > 2 else 1 - buy_prob - sell_prob
                df["buy_proba"] = buy_prob
                df["sell_proba"] = sell_prob
                df["no_trade_proba"] = no_trade_prob
            df["signal"] = self.predict_signal(X)
            df["signal_proba"] = np.where(
                df["signal"] == 1,
                df["buy_proba"],
                np.where(df["signal"] == -1, df["sell_proba"], df["no_trade_proba"]),
            )
        else:
            df["signal_proba"] = self.predict_proba(X)
            df["signal"] = self.predict_signal(X)
        return df

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self, df: pd.DataFrame, feature_columns: List[str], n_splits: int = 5
    ) -> List[Dict[str, float]]:
        """Walk-forward cross-validation using ``TimeSeriesSplit``."""
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        y = df["target"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            fold_scaler = StandardScaler()
            fold_scaler.fit(X[train_idx])
            X_train = fold_scaler.transform(X[train_idx])
            X_validation = fold_scaler.transform(X[test_idx])
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
            model.fit(X_train, y[train_idx], verbose=False)
            metrics = self._evaluate(X_validation, y[test_idx], model=model)
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
        probabilities = m.predict_proba(X)
        unique_labels = np.unique(y)
        is_binary_problem = probabilities.ndim == 2 and probabilities.shape[1] == 2 and len(unique_labels) <= 2

        if is_binary_problem:
            y_proba = probabilities[:, 1]
            roc_auc = roc_auc_score(y, y_proba)
            precision = precision_score(y, y_pred, average="binary", zero_division=0)
            recall = recall_score(y, y_pred, average="binary", zero_division=0)
            f1 = f1_score(y, y_pred, average="binary", zero_division=0)
        else:
            try:
                roc_auc = roc_auc_score(y, probabilities, multi_class="ovr")
            except ValueError:
                roc_auc = float("nan")
            precision = precision_score(y, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        }

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
