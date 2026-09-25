"""Event-driven backtest on minute-accurate label outcomes.

Each LONG/SHORT signal at bar T enters at the open of the first minute after the bar closes
(T + 1h) and exits where the triple-barrier label says: at TP, at SL, or at the last close
of the horizon, after the labelled number of minutes. Rules:

- one open position per symbol (signals while a position is open are skipped)
- fees and slippage on both sides (decision.net_return)
- fixed fractional sizing: a stop-out loses ``risk_per_trade`` of equity before costs,
  capped at ``max_position_leverage`` x equity notional
- after realized losses of ``max_daily_loss`` of the day's starting equity, no new trades
  that UTC day

    python -m src.backtest --model xgb      # walk-forward OOF -> calibrate -> decide -> trades CSV
"""
from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, ensure_dir, load_config, resolve_path
from src.decision import LONG, NAMES, net_return
from src.labels import LOSS, WIN

EXIT_REASON = {WIN: "TP", LOSS: "SL"}
TRADE_COLS = ["symbol", "side", "signal_time", "entry_time", "exit_time", "entry", "tp", "sl", "exit",
              "exit_reason", "minutes", "confidence", "notional", "gross_ret", "net_ret", "pnl",
              "equity_at_entry"]


@dataclass
class BacktestResult:
    trades: pd.DataFrame          # one row per executed trade
    equity: pd.Series             # realized equity on an hourly grid
    skipped: dict[str, int]       # signals not taken, by reason
    start: pd.Timestamp
    end: pd.Timestamp


def _side_cols(side: int) -> str:
    return "long" if side == LONG else "short"


def backtest(signals: pd.DataFrame, cfg: Config, start: pd.Timestamp | None = None,
             end: pd.Timestamp | None = None) -> BacktestResult:
    """Run the backtest. ``signals`` needs symbol, time, decision, entry, atr and per-side
    ``{side}_exit``, ``{side}_minutes``, ``{side}_label`` columns (``confidence`` optional)."""
    bt, lc = cfg.backtest, cfg.labels
    sig = signals[signals["decision"] != 0].sort_values(["time", "symbol"])
    equity = bt.initial_equity
    open_until: dict[str, pd.Timestamp] = {}
    pending: list[tuple[pd.Timestamp, int, float]] = []   # (exit time, trade id, pnl)
    day_start: dict = {}
    day_pnl: dict = {}
    trades, skipped = [], {"position_open": 0, "daily_loss_stop": 0}

    def settle(until: pd.Timestamp) -> None:
        nonlocal equity
        while pending and pending[0][0] <= until:
            t, _, pnl = heapq.heappop(pending)
            d = t.date()
            day_start.setdefault(d, equity)
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            equity += pnl

    for r in sig.itertuples(index=False):
        side = int(r.decision)
        name = _side_cols(side)
        t_in = r.time + pd.Timedelta(hours=1)
        settle(t_in)
        if open_until.get(r.symbol, t_in) > t_in:
            skipped["position_open"] += 1
            continue
        d = t_in.date()
        day_start.setdefault(d, equity)
        if day_pnl.get(d, 0.0) <= -bt.max_daily_loss * day_start[d]:
            skipped["daily_loss_stop"] += 1
            continue

        entry, atr = float(r.entry), float(r.atr)
        exit_px = float(getattr(r, f"{name}_exit"))
        minutes = int(getattr(r, f"{name}_minutes"))
        label = int(getattr(r, f"{name}_label"))
        sl_frac = lc.sl_atr * atr / entry
        notional = min(bt.risk_per_trade * equity / sl_frac, bt.max_position_leverage * equity)
        net = float(net_return(side, entry, exit_px, cfg))
        pnl = notional * net
        t_out = t_in + pd.Timedelta(minutes=minutes)
        heapq.heappush(pending, (t_out, len(trades), pnl))
        open_until[r.symbol] = t_out
        trades.append({
            "symbol": r.symbol, "side": NAMES[side], "signal_time": r.time, "entry_time": t_in,
            "exit_time": t_out, "entry": entry, "tp": entry + side * lc.tp_atr * atr,
            "sl": entry - side * lc.sl_atr * atr, "exit": exit_px,
            "exit_reason": EXIT_REASON.get(label, "TIMEOUT"), "minutes": minutes,
            "confidence": getattr(r, "confidence", np.nan), "notional": notional,
            "gross_ret": side * (exit_px / entry - 1), "net_ret": net, "pnl": pnl,
            "equity_at_entry": equity,
        })
    settle(pd.Timestamp.max.tz_localize("UTC"))

    tr = pd.DataFrame(trades, columns=TRADE_COLS)
    start = start or (signals["time"].min() + pd.Timedelta(hours=1))
    last_exit = tr["exit_time"].max() if len(tr) else start
    end = end or max(signals["time"].max() + pd.Timedelta(hours=1), last_exit)
    return BacktestResult(tr, equity_curve(tr, start, end, bt.initial_equity), skipped, start, end)


def equity_curve(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, initial: float) -> pd.Series:
    """Realized equity at the end of each hour (P/L booked at exit)."""
    grid = pd.date_range(start.floor("h"), end.ceil("h"), freq="1h")
    if trades.empty:
        return pd.Series(initial, index=grid, name="equity")
    pnl = trades.groupby(trades["exit_time"].dt.ceil("h"))["pnl"].sum()
    return (initial + pnl.reindex(grid, fill_value=0.0).cumsum()).rename("equity")


def save_trades(res: BacktestResult, path) -> None:
    ensure_dir(path.parent)
    res.trades.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S%z")
