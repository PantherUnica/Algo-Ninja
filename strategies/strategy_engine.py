"""
strategy_engine.py
--------------------
Defines 12+ trading strategies that combine the Random Forest return
prediction with EMA7 / RSI8 / ADX9 filters. Each strategy is a function
that takes the feature/prediction DataFrame and returns a Series of
positions: 1 = long, -1 = short, 0 = flat.

Strategies are grouped into families:
  1-3   RF-signal only, varying confidence thresholds
  4-6   RF + RSI8 filter (avoid overbought/oversold traps)
  7-9   RF + ADX9 trend-strength filter (trade only trending regimes)
  10-12 RF + EMA7 trend-alignment filter
  13    Combined "ensemble vote" strategy (all filters must agree)
"""

from typing import Callable, Dict

import pandas as pd


def _base_signal(pred_return: pd.Series, threshold: float) -> pd.Series:
    sig = pd.Series(0, index=pred_return.index)
    sig[pred_return > threshold] = 1
    sig[pred_return < -threshold] = -1
    return sig


# ---------------------------------------------------------------- family 1
def strategy_rf_conservative(df: pd.DataFrame) -> pd.Series:
    """RF only, high confidence threshold (0.5%)."""
    return _base_signal(df["predicted_return"], threshold=0.005)


def strategy_rf_moderate(df: pd.DataFrame) -> pd.Series:
    """RF only, medium confidence threshold (0.2%)."""
    return _base_signal(df["predicted_return"], threshold=0.002)


def strategy_rf_aggressive(df: pd.DataFrame) -> pd.Series:
    """RF only, low confidence threshold (0.05%) — trades more often."""
    return _base_signal(df["predicted_return"], threshold=0.0005)


# ---------------------------------------------------------------- family 2 (RSI8 filter)
def strategy_rf_rsi_meanreversion(df: pd.DataFrame) -> pd.Series:
    """Long only when RF bullish AND RSI8 < 40 (not overbought)."""
    sig = _base_signal(df["predicted_return"], threshold=0.002)
    sig[(sig == 1) & (df["RSI8"] >= 40)] = 0
    sig[(sig == -1) & (df["RSI8"] <= 60)] = 0
    return sig


def strategy_rf_rsi_momentum(df: pd.DataFrame) -> pd.Series:
    """Long only when RF bullish AND RSI8 > 55 (momentum confirmation)."""
    sig = _base_signal(df["predicted_return"], threshold=0.002)
    sig[(sig == 1) & (df["RSI8"] <= 55)] = 0
    sig[(sig == -1) & (df["RSI8"] >= 45)] = 0
    return sig


def strategy_rf_rsi_extreme_fade(df: pd.DataFrame) -> pd.Series:
    """Fade extremes: short if RSI8 > 80, long if RSI8 < 20, regardless of RF."""
    sig = pd.Series(0, index=df.index)
    sig[df["RSI8"] < 20] = 1
    sig[df["RSI8"] > 80] = -1
    return sig


# ---------------------------------------------------------------- family 3 (ADX9 filter)
def strategy_rf_adx_trend_only(df: pd.DataFrame) -> pd.Series:
    """Trade RF signal only when ADX9 > 25 (strong trend)."""
    sig = _base_signal(df["predicted_return"], threshold=0.002)
    sig[df["ADX9"] <= 25] = 0
    return sig


def strategy_rf_adx_strong_trend(df: pd.DataFrame) -> pd.Series:
    """Stricter version: ADX9 > 35."""
    sig = _base_signal(df["predicted_return"], threshold=0.002)
    sig[df["ADX9"] <= 35] = 0
    return sig


def strategy_rf_adx_range_bound(df: pd.DataFrame) -> pd.Series:
    """Opposite regime: trade only when ADX9 < 20 (range-bound market)."""
    sig = _base_signal(df["predicted_return"], threshold=0.001)
    sig[df["ADX9"] >= 20] = 0
    return sig


# ---------------------------------------------------------------- family 4 (EMA7 alignment)
def strategy_rf_ema_trend_align(df: pd.DataFrame) -> pd.Series:
    """Only go long if Close > EMA7 and RF bullish (trend alignment)."""
    sig = _base_signal(df["predicted_return"], threshold=0.002)
    sig[(sig == 1) & (df["Close"] <= df["EMA7"])] = 0
    sig[(sig == -1) & (df["Close"] >= df["EMA7"])] = 0
    return sig


def strategy_rf_ema_slope_confirm(df: pd.DataFrame) -> pd.Series:
    """Require EMA7 slope to agree with RF direction."""
    sig = _base_signal(df["predicted_return"], threshold=0.002)
    sig[(sig == 1) & (df["EMA7_slope"] <= 0)] = 0
    sig[(sig == -1) & (df["EMA7_slope"] >= 0)] = 0
    return sig


def strategy_rf_ema_pullback(df: pd.DataFrame) -> pd.Series:
    """Buy dips: Close_to_EMA7 slightly negative but RF bullish (pullback-in-uptrend)."""
    sig = pd.Series(0, index=df.index)
    bullish = df["predicted_return"] > 0.001
    pullback = (df["Close_to_EMA7"] < 0) & (df["Close_to_EMA7"] > -0.02)
    sig[bullish & pullback] = 1
    return sig


# ---------------------------------------------------------------- family 5 (ensemble)
def strategy_ensemble_vote(df: pd.DataFrame) -> pd.Series:
    """
    All three filters (RF direction, RSI8 not extreme, ADX9 trending)
    must agree — highest conviction, lowest trade frequency.
    """
    sig = _base_signal(df["predicted_return"], threshold=0.0015)
    trending = df["ADX9"] > 20
    rsi_ok_long = df["RSI8"] < 65
    rsi_ok_short = df["RSI8"] > 35

    final = pd.Series(0, index=df.index)
    long_mask = (sig == 1) & trending & rsi_ok_long
    short_mask = (sig == -1) & trending & rsi_ok_short
    final[long_mask] = 1
    final[short_mask] = -1
    return final


def strategy_ensemble_weighted(df: pd.DataFrame) -> pd.Series:
    """
    Score-based ensemble: +1/-1/0 vote from each of RF, RSI8, ADX9-slope
    proxy; trade only when the summed score is unanimous-ish (>=2).
    """
    rf_vote = pd.Series(0, index=df.index)
    rf_vote[df["predicted_return"] > 0.001] = 1
    rf_vote[df["predicted_return"] < -0.001] = -1

    rsi_vote = pd.Series(0, index=df.index)
    rsi_vote[df["RSI8"] > 50] = 1
    rsi_vote[df["RSI8"] < 50] = -1

    ema_vote = pd.Series(0, index=df.index)
    ema_vote[df["Close"] > df["EMA7"]] = 1
    ema_vote[df["Close"] < df["EMA7"]] = -1

    score = rf_vote + rsi_vote + ema_vote
    sig = pd.Series(0, index=df.index)
    sig[score >= 2] = 1
    sig[score <= -2] = -1
    return sig


# Registry of all strategies -> used by the backtester / CLI
STRATEGY_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "rf_conservative": strategy_rf_conservative,
    "rf_moderate": strategy_rf_moderate,
    "rf_aggressive": strategy_rf_aggressive,
    "rf_rsi_meanreversion": strategy_rf_rsi_meanreversion,
    "rf_rsi_momentum": strategy_rf_rsi_momentum,
    "rf_rsi_extreme_fade": strategy_rf_rsi_extreme_fade,
    "rf_adx_trend_only": strategy_rf_adx_trend_only,
    "rf_adx_strong_trend": strategy_rf_adx_strong_trend,
    "rf_adx_range_bound": strategy_rf_adx_range_bound,
    "rf_ema_trend_align": strategy_rf_ema_trend_align,
    "rf_ema_slope_confirm": strategy_rf_ema_slope_confirm,
    "rf_ema_pullback": strategy_rf_ema_pullback,
    "ensemble_vote": strategy_ensemble_vote,
    "ensemble_weighted": strategy_ensemble_weighted,
}
