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


def random_baseline(pool: pd.DataFrame, n_trades: int, cfg: Config, start: pd.Timestamp,
                    end: pd.Timestamp, sims: int | None = None, seed: int | None = None) -> tuple[dict, list[float]]:
    """Random entries with the same trade count. ``pool`` holds candidate rows with market outcomes.

    Returns (median-simulation style summary with percentiles, list of net returns per simulation).
    """
    sims = sims or cfg.metrics.random_sims
    rng = np.random.default_rng(cfg.model.seed if seed is None else seed)
    if n_trades == 0 or pool.empty:
        return {"trades": 0, "net_return": 0.0, "net_return_p05": 0.0, "net_return_p95": 0.0,
                "mean_trade_ret": np.nan, "sharpe": 0.0, "max_drawdown": 0.0}, []
    rets, rows = [], []
    for _ in range(sims):
        pick = pool.iloc[np.sort(rng.choice(len(pool), min(n_trades, len(pool)), replace=False))].copy()
        pick["decision"] = rng.choice([LONG, SHORT], len(pick))
        res = backtest(pick, cfg, start, end)
        s = summarize(res, cfg)
        rets.append(s["net_return"])
        rows.append(s)
    df = pd.DataFrame(rows)
    return {"trades": float(df["trades"].mean()), "net_return": float(df["net_return"].median()),
            "net_return_p05": float(np.quantile(rets, 0.05)), "net_return_p95": float(np.quantile(rets, 0.95)),
            "mean_trade_ret": float(df["mean_trade_ret"].mean()), "sharpe": float(df["sharpe"].median()),
            "max_drawdown": float(df["max_drawdown"].median()), "win_rate": float(df["win_rate"].mean()),
            "profit_factor": float(df["profit_factor"].replace(np.inf, np.nan).median()),
            "sortino": float(df["sortino"].replace(np.inf, np.nan).median()),
            "exposure": float(df["exposure"].mean())}, rets


def _fmt(k: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if k in ("trades",):
        return f"{v:,.0f}"
    if k in ("net_return", "net_return_p05", "net_return_p95", "win_rate", "max_drawdown", "exposure"):
        return f"{v:+.2%}" if k.startswith("net") or k == "max_drawdown" else f"{v:.1%}"
    if k in ("mean_trade_ret", "avg_win", "avg_loss"):
        return f"{v:+.3%}"
    return f"{v:.2f}"


COLUMNS = [("net_return", "Net return"), ("trades", "Trades"), ("win_rate", "Win rate"),
           ("avg_win", "Avg win"), ("avg_loss", "Avg loss"), ("profit_factor", "Profit factor"),
           ("mean_trade_ret", "Mean trade"), ("max_drawdown", "Max DD"), ("sharpe", "Sharpe"),
           ("sortino", "Sortino"), ("exposure", "Exposure")]


def metrics_table(rows: dict[str, dict]) -> str:
    """Markdown table: one row per strategy / baseline."""
    head = "| Strategy | " + " | ".join(n for _, n in COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(COLUMNS)
    body = [f"| {name} | " + " | ".join(_fmt(k, m.get(k)) for k, _ in COLUMNS) + " |" for name, m in rows.items()]
    return "\n".join([head, sep, *body])


def plot_equity(curves: dict[str, pd.Series], path, title: str) -> None:
    """Equity curves (normalized to 1) with a drawdown panel."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, eq in curves.items():
        a1.plot(eq.index, eq / eq.iloc[0], label=name, lw=1.2)
        a2.plot(eq.index, drawdown(eq) * 100, lw=1)
    a1.set_ylabel("equity (start = 1)")
    a1.set_title(title)
    a1.legend(loc="upper left", fontsize=8)
    a1.grid(alpha=0.3)
    a2.set_ylabel("drawdown %")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
