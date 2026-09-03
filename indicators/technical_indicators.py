"""
technical_indicators.py
------------------------
Core technical indicators used as model features:
    - EMA7  : 7-period Exponential Moving Average
    - RSI8  : 8-period Relative Strength Index
    - ADX9  : 9-period Average Directional Index (trend strength)

All functions operate on a DataFrame with columns: Open, High, Low, Close, Volume
and return pandas Series aligned to the same index.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 8) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).
    RSI = 100 - (100 / (1 + RS)), RS = avg_gain / avg_loss
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)  # neutral fill for warmup period


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def adx(df: pd.DataFrame, period: int = 9) -> pd.Series:
    """
    Average Directional Index — measures trend strength (not direction).
    Uses Wilder's smoothing on +DM / -DM / True Range.
    """
    high, low = df["High"], df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx_val.fillna(0)


def add_all_indicators(
    df: pd.DataFrame,
    ema_span: int = 7,
    rsi_period: int = 8,
    adx_period: int = 9,
) -> pd.DataFrame:
    """
    Returns a copy of df with EMA7, RSI8, ADX9 (and a few helper features)
    appended as columns. Rows containing warmup-period NaNs are dropped.
    """
    out = df.copy()
    out["EMA7"] = ema(out["Close"], ema_span)
    out["RSI8"] = rsi(out["Close"], rsi_period)
    out["ADX9"] = adx(out, adx_period)

    # Helper / derived features that improve the RF model
    out["EMA7_slope"] = out["EMA7"].diff()
    out["Close_to_EMA7"] = (out["Close"] - out["EMA7"]) / out["EMA7"]
    out["Return_1d"] = out["Close"].pct_change()
    out["Volatility_5d"] = out["Return_1d"].rolling(5).std()

    out = out.dropna()
    return out
