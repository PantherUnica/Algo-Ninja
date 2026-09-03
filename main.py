"""
main.py
--------
End-to-end pipeline:
  1. Fetch OHLCV data
  2. Compute EMA7 / RSI8 / ADX9 + derived features
  3. Chronological train/test split
  4. Fit Random Forest regressor on training set
  5. Generate predictions on test set
  6. Run all 12+ strategies through the backtester
  7. Print leaderboard + best strategy's equity curve stats

Usage
-----
    python main.py --ticker RELIANCE.NS --period 2y --interval 1d

If you don't have internet access / yfinance installed, point --csv at a
local OHLCV file instead:

    python main.py --csv data/sample_ohlcv.csv
"""

import argparse
import sys

import pandas as pd

from data.data_loader import fetch_ohlcv, load_csv, train_test_split_series
from indicators.technical_indicators import add_all_indicators
from models.rf_predictor import RandomForestPredictor, RFPredictorConfig
from strategies.strategy_engine import STRATEGY_REGISTRY
from backtest.backtester import Backtester, run_all_strategies


def parse_args():
    p = argparse.ArgumentParser(description="Algo Ninja — RF + Technical Indicator Trading Pipeline")
    p.add_argument("--ticker", type=str, default=None, help="Ticker symbol, e.g. RELIANCE.NS")
    p.add_argument("--csv", type=str, default=None, help="Path to local OHLCV CSV (alternative to --ticker)")
    p.add_argument("--period", type=str, default="2y")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--horizon", type=int, default=1, help="Bars ahead to predict")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--cost-bps", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.ticker and not args.csv:
        print("ERROR: provide either --ticker or --csv", file=sys.stderr)
        sys.exit(1)

    # 1. Load data
    if args.csv:
        print(f"Loading data from CSV: {args.csv}")
        raw = load_csv(args.csv)
    else:
        print(f"Fetching {args.ticker} | period={args.period} interval={args.interval}")
        raw = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)

    print(f"Loaded {len(raw)} bars: {raw.index[0].date()} -> {raw.index[-1].date()}")

    # 2. Feature engineering
    featured = add_all_indicators(raw, ema_span=7, rsi_period=8, adx_period=9)
    print(f"After indicator warmup: {len(featured)} usable bars")

    # 3. Chronological split
    train, test = train_test_split_series(featured, test_size=args.test_size)
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")

    # 4. Fit RF model
    config = RFPredictorConfig(horizon=args.horizon)
    model = RandomForestPredictor(config)
    model.fit(train)

    metrics = model.evaluate(test)
    print("\n--- Model Evaluation (held-out test set) ---")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.4f}")

    print("\n--- Feature Importances ---")
    print(model.feature_importances().to_string())

    # 5. Predictions on test set
    test = test.copy()
    test["predicted_return"] = model.predict(test)

    # 6. Backtest every strategy
    bt = Backtester(transaction_cost_bps=args.cost_bps, initial_capital=args.capital)
    leaderboard = run_all_strategies(test, STRATEGY_REGISTRY, bt)

    print(f"\n--- Strategy Leaderboard ({len(STRATEGY_REGISTRY)} strategies) ---")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["strategy"]
    best_result = bt.run(test, STRATEGY_REGISTRY[best_name], strategy_name=best_name)

    print(f"\nBest strategy: {best_name}")
    print(f"  Profitability : {best_result.profitability_pct:.2f}%")
    print(f"  Total return  : {best_result.total_return_pct:.2f}%")
    print(f"  Sharpe ratio  : {best_result.sharpe_ratio:.2f}")
    print(f"  Max drawdown  : {best_result.max_drawdown_pct:.2f}%")
    print(f"  # of trades   : {best_result.num_trades}")

    out_path = "outputs/leaderboard.csv"
    leaderboard.to_csv(out_path, index=False)
    print(f"\nLeaderboard saved to {out_path}")


if __name__ == "__main__":
    main()
