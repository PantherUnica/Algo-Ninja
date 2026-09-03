"""
backtester.py
---------------
A simple, vectorized, long/short backtesting engine with:
    - transaction cost & slippage modelling
    - equity curve, drawdown, Sharpe, win-rate, profitability %
    - support for running every registered strategy and ranking them
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    total_return_pct: float
    profitability_pct: float  # % of trades that were profitable
    win_rate_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "total_return_%": round(self.total_return_pct, 2),
            "profitability_%": round(self.profitability_pct, 2),
            "win_rate_%": round(self.win_rate_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_%": round(self.max_drawdown_pct, 2),
            "num_trades": self.num_trades,
        }


class Backtester:
    def __init__(
        self,
        transaction_cost_bps: float = 5.0,  # 0.05% per side (brokerage + slippage)
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.06,  # annualized, for Sharpe
        bars_per_year: int = 252,
    ):
        self.cost = transaction_cost_bps / 10_000
        self.capital = initial_capital
        self.rf_rate = risk_free_rate
        self.bars_per_year = bars_per_year

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], pd.Series],
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        positions = strategy_fn(df).reindex(df.index).fillna(0)
        returns = df["Close"].pct_change().fillna(0)

        # Position taken at bar t generates return realized at bar t+1
        strat_returns = positions.shift(1).fillna(0) * returns

        # Transaction costs applied whenever position changes
        turnover = positions.diff().abs().fillna(0)
        costs = turnover * self.cost
        net_returns = strat_returns - costs

        equity_curve = self.capital * (1 + net_returns).cumprod()

        trade_log = self._build_trade_log(df, positions, net_returns)

        total_return_pct = (equity_curve.iloc[-1] / self.capital - 1) * 100
        sharpe = self._sharpe(net_returns)
        max_dd = self._max_drawdown(equity_curve)

        if len(trade_log) > 0:
            profitable = (trade_log["pnl_pct"] > 0).sum()
            profitability_pct = 100 * profitable / len(trade_log)
            win_rate_pct = profitability_pct
        else:
            profitability_pct = 0.0
            win_rate_pct = 0.0

        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity_curve,
            trade_log=trade_log,
            total_return_pct=total_return_pct,
            profitability_pct=profitability_pct,
            win_rate_pct=win_rate_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(trade_log),
        )

    @staticmethod
    def _build_trade_log(df: pd.DataFrame, positions: pd.Series, net_returns: pd.Series) -> pd.DataFrame:
        """Collapse consecutive same-position bars into discrete trades."""
        trades = []
        pos = positions.values
        idx = positions.index
        in_trade = False
        entry_i = None
        direction = 0

        for i in range(1, len(pos)):
            if not in_trade and pos[i - 1] != 0:
                in_trade = True
                entry_i = i - 1
                direction = pos[i - 1]
            if in_trade and (pos[i] != direction):
                exit_i = i
                pnl_pct = direction * (df["Close"].iloc[exit_i] / df["Close"].iloc[entry_i] - 1) * 100
                trades.append(
                    {
                        "entry_date": idx[entry_i],
                        "exit_date": idx[exit_i],
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "pnl_pct": pnl_pct,
                    }
                )
                in_trade = False
                if pos[i] != 0:
                    in_trade = True
                    entry_i = i
                    direction = pos[i]

        return pd.DataFrame(trades)

    def _sharpe(self, net_returns: pd.Series) -> float:
        if net_returns.std() == 0:
            return 0.0
        daily_rf = self.rf_rate / self.bars_per_year
        excess = net_returns - daily_rf
        return float(np.sqrt(self.bars_per_year) * excess.mean() / net_returns.std())

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return float(drawdown.min() * 100)


def run_all_strategies(
    df: pd.DataFrame,
    registry: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    backtester: Backtester = None,
) -> pd.DataFrame:
    """
    Runs every strategy in `registry` against `df` and returns a leaderboard
    DataFrame sorted by profitability_% descending.
    """
    bt = backtester or Backtester()
    rows = []
    for name, fn in registry.items():
        result = bt.run(df, fn, strategy_name=name)
        rows.append(result.summary())
    leaderboard = pd.DataFrame(rows).sort_values("profitability_%", ascending=False).reset_index(drop=True)
    return leaderboard
