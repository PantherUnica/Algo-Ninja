"""
rf_predictor.py
-----------------
Random Forest Regression model that predicts next-period price movement
(return) using EMA7 / RSI8 / ADX9 + derived technical features.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # older scikit-learn versions
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5

DEFAULT_FEATURES: List[str] = [
    "EMA7",
    "RSI8",
    "ADX9",
    "EMA7_slope",
    "Close_to_EMA7",
    "Return_1d",
    "Volatility_5d",
]


@dataclass
class RFPredictorConfig:
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 5
    random_state: int = 42
    horizon: int = 1  # predict return `horizon` bars ahead
    features: List[str] = field(default_factory=lambda: DEFAULT_FEATURES)


class RandomForestPredictor:
    """
    Wraps a RandomForestRegressor to predict forward returns from
    technical-indicator features, and exposes helpers to turn those
    predictions into a directional trading signal.
    """

    def __init__(self, config: RFPredictorConfig = None):
        self.config = config or RFPredictorConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """Forward return over `horizon` bars — the regression target."""
        h = self.config.horizon
        target = df["Close"].shift(-h) / df["Close"] - 1
        return target

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[self.config.features].copy()
        y = self.build_target(df)
        valid = y.notna()
        return X[valid], y[valid]

    def fit(self, df: pd.DataFrame) -> "RandomForestPredictor":
        X, y = self.prepare_dataset(df)
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Model must be fit() before predict().")
        X = df[self.config.features]
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index, name="predicted_return")

    def evaluate(self, df: pd.DataFrame) -> dict:
        X, y = self.prepare_dataset(df)
        preds = self.model.predict(X)
        return {
            "MAE": mean_absolute_error(y, preds),
            "RMSE": root_mean_squared_error(y, preds),
            "R2": r2_score(y, preds),
            "directional_accuracy": float(np.mean(np.sign(preds) == np.sign(y))),
        }

    def feature_importances(self) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Model must be fit() before inspecting importances.")
        return pd.Series(
            self.model.feature_importances_, index=self.config.features
        ).sort_values(ascending=False)
