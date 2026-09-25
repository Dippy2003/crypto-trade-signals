"""Performance metrics, plots and baselines.

Sharpe and Sortino use hourly returns of the realized equity curve, annualized with
sqrt(8760) because crypto trades 24/7. Every strategy is compared with:
  - buy-and-hold (equal weight across symbols, one round trip of costs)
  - random entries: the same number of trades, random time and side, the same TP/SL
    outcomes, sizing and costs, run through the same backtester
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

from src.backtest import BacktestResult, backtest
from src.config import Config
from src.decision import LONG, SHORT, round_trip_cost


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall as a (negative) fraction."""
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


COLUMNS = [("net_return", "Net return"), ("trades", "Trades"), ("win_rate", "Win rate"),
           ("avg_win", "Avg win"), ("avg_loss", "Avg loss"), ("profit_factor", "Profit factor"),
           ("mean_trade_ret", "Mean trade"), ("max_drawdown", "Max DD"), ("sharpe", "Sharpe"),
           ("sortino", "Sortino"), ("exposure", "Exposure")]
