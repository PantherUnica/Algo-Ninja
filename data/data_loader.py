"""
data_loader.py
----------------
Fetches historical / live OHLCV stock data using yfinance and prepares
it for the feature-engineering and modeling pipeline.
"""

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. "RELIANCE.NS", "AAPL".
    period : str
        Lookback window (e.g. "1mo", "3mo", "6mo", "1y", "2y", "max").
    interval : str
        Bar interval (e.g. "1d", "1h", "15m").

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume  (DatetimeIndex)
    """
    if yf is None:
        raise ImportError(
            "yfinance is required. Install with: pip install yfinance"
        )

    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def load_csv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a local CSV (must contain Date, Open, High, Low,
    Close, Volume columns). Useful for offline backtesting on saved data.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.sort_index()


def train_test_split_series(df: pd.DataFrame, test_size: float = 0.2):
    """
    Chronological (non-shuffled) split — critical for time series data
    to avoid lookahead bias.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
