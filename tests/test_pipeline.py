"""
test_pipeline.py
-------------------
Basic sanity tests for indicators, RF model, strategies, and backtester.
Run with: pytest tests/
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indicators.technical_indicators import ema, rsi, adx, add_all_indicators
from models.rf_predictor import RandomForestPredictor, RFPredictorConfig
from strategies.strategy_engine import STRATEGY_REGISTRY
from backtest.backtester import Backtester, run_all_strategies


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    returns = np.random.normal(0.0003, 0.012, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.0
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_ema_length_matches_input(sample_df):
    result = ema(sample_df["Close"], span=7)
    assert len(result) == len(sample_df)


def test_rsi_bounds(sample_df):
    result = rsi(sample_df["Close"], period=8)
    assert result.min() >= 0
    assert result.max() <= 100


def test_adx_non_negative(sample_df):
    result = adx(sample_df, period=9)
    assert (result >= 0).all()


def test_add_all_indicators_columns(sample_df):
    featured = add_all_indicators(sample_df)
    for col in ["EMA7", "RSI8", "ADX9", "EMA7_slope", "Close_to_EMA7", "Return_1d", "Volatility_5d"]:
        assert col in featured.columns
    assert not featured.isna().any().any()


def test_rf_predictor_fits_and_predicts(sample_df):
    featured = add_all_indicators(sample_df)
    split = int(len(featured) * 0.8)
    train, test = featured.iloc[:split], featured.iloc[split:]

    model = RandomForestPredictor(RFPredictorConfig(n_estimators=50))
    model.fit(train)
    preds = model.predict(test)

    assert len(preds) == len(test)
    assert preds.notna().all()


def test_at_least_twelve_strategies_registered():
    assert len(STRATEGY_REGISTRY) >= 12


def test_backtester_runs_all_strategies(sample_df):
    featured = add_all_indicators(sample_df)
    featured["predicted_return"] = np.random.normal(0, 0.005, len(featured))

    bt = Backtester()
    leaderboard = run_all_strategies(featured, STRATEGY_REGISTRY, bt)

    assert len(leaderboard) == len(STRATEGY_REGISTRY)
    assert "profitability_%" in leaderboard.columns
    assert (leaderboard["profitability_%"] >= 0).all()
    assert (leaderboard["profitability_%"] <= 100).all()


def test_single_strategy_backtest_result_shape(sample_df):
    featured = add_all_indicators(sample_df)
    featured["predicted_return"] = np.random.normal(0, 0.005, len(featured))

    bt = Backtester()
    result = bt.run(featured, STRATEGY_REGISTRY["rf_conservative"], "rf_conservative")

    assert len(result.equity_curve) == len(featured)
    assert isinstance(result.summary(), dict)
