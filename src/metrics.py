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


def sharpe_sortino(equity: pd.Series, periods_per_year: int) -> tuple[float, float]:
    r = equity.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0, 0.0
    ann = np.sqrt(periods_per_year)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    sortino = float(r.mean() / downside * ann) if downside > 0 else float("inf")
    return float(r.mean() / r.std() * ann), sortino


def exposure(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Fraction of the period with at least one open position."""
    total = (end - start).total_seconds()
    if trades.empty or total <= 0:
        return 0.0
    iv = trades[["entry_time", "exit_time"]].sort_values("entry_time").to_numpy()
    covered, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            covered += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += (cur_e - cur_s).total_seconds()
    return min(covered / total, 1.0)


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
                "profit_factor": np.nan, "mean_trade_ret": np.nan}
    r = trades["net_ret"]
    wins, losses = trades.loc[r > 0, "pnl"], trades.loc[r <= 0, "pnl"]
    return {
        "trades": len(trades),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(r[r > 0].mean()) if (r > 0).any() else np.nan,
        "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "mean_trade_ret": float(r.mean()),
    }


def summarize(res: BacktestResult, cfg: Config) -> dict:
    eq = res.equity
    sharpe, sortino = sharpe_sortino(eq, cfg.metrics.periods_per_year)
    return {
        "net_return": float(eq.iloc[-1] / cfg.backtest.initial_equity - 1),
        **trade_stats(res.trades),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe, "sortino": sortino,
        "exposure": exposure(res.trades, res.start, res.end),
    }


def buy_and_hold(closes: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp, cfg: Config) -> tuple[dict, pd.Series]:
    """Equal-weight buy-and-hold of every symbol over [start, end], paying one round trip."""
    grid = pd.date_range(start.floor("h"), end.ceil("h"), freq="1h")
    rel = []
    for c in closes.values():
        s = c.reindex(grid).ffill().bfill()
        rel.append(s / s.iloc[0])
    eq = cfg.backtest.initial_equity * (1 - round_trip_cost(cfg)) * pd.concat(rel, axis=1).mean(axis=1)
    sharpe, sortino = sharpe_sortino(eq, cfg.metrics.periods_per_year)
    stats = {"net_return": float(eq.iloc[-1] / cfg.backtest.initial_equity - 1), "trades": len(closes),
             "win_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan, "profit_factor": np.nan,
             "mean_trade_ret": np.nan, "max_drawdown": max_drawdown(eq), "sharpe": sharpe,
             "sortino": sortino, "exposure": 1.0}
    return stats, eq.rename("equity")


COLUMNS = [("net_return", "Net return"), ("trades", "Trades"), ("win_rate", "Win rate"),
           ("avg_win", "Avg win"), ("avg_loss", "Avg loss"), ("profit_factor", "Profit factor"),
           ("mean_trade_ret", "Mean trade"), ("max_drawdown", "Max DD"), ("sharpe", "Sharpe"),
           ("sortino", "Sortino"), ("exposure", "Exposure")]
