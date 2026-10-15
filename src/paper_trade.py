"""Paper trading: log every hourly signal and its eventual outcome, then compare with the backtest.

Each run (1) resolves open paper trades from 1m candles using the same barrier rules as the
labels (entry = open of the first minute after the bar closes, TP/SL from that entry and the
bar's ATR, both in one minute = SL, timeout after the horizon), then (2) logs the newest signal
per symbol. Like the backtest, a new LONG/SHORT while that symbol has an open paper trade is
logged as SKIPPED.

Log: reports/paper_trades.csv   Summary: reports/paper_summary.md

    python -m src.paper_trade --once        # one cycle (use with a scheduler)
    python -m src.paper_trade --loop        # run every hour
    python -m src.paper_trade --summary     # paper vs backtest
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import Config, ensure_dir, load_config, resolve_path
from src.decision import net_return
from src.labels import LOSS, NO_TRADE, WIN, barrier_outcome
from src.live import fetch_minutes, get_signal

LOG_COLS = ["timestamp", "bar_time", "symbol", "model", "decision", "status", "probability",
            "p_long_win", "p_short_win", "atr", "entry", "tp", "sl", "exit", "exit_time",
            "exit_reason", "minutes", "gross_ret", "pnl"]
TIME_COLS = ["timestamp", "bar_time", "exit_time"]
TEXT_COLS = ["symbol", "model", "decision", "status", "exit_reason"]
SIDE = {"LONG": 1, "SHORT": -1}
REASON = {WIN: "TP", LOSS: "SL", NO_TRADE: "TIMEOUT"}


def log_path(cfg: Config):
    return resolve_path(cfg, "reports") / cfg.paper.log_file


def typed(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed dtypes for every log column (empty columns would otherwise be read as float)."""
    df = df.reindex(columns=LOG_COLS)
    for c in LOG_COLS:
        if c in TIME_COLS:
            df[c] = pd.to_datetime(df[c], utc=True)
        elif c in TEXT_COLS:
            df[c] = df[c].astype(object)
        else:
            df[c] = pd.to_numeric(df[c]).astype("float64")
    return df


def load_log(cfg: Config) -> pd.DataFrame:
    p = log_path(cfg)
    return typed(pd.read_csv(p) if p.exists() else pd.DataFrame(columns=LOG_COLS))


def save_log(cfg: Config, log: pd.DataFrame) -> None:
    ensure_dir(log_path(cfg).parent)
    log[LOG_COLS].to_csv(log_path(cfg), index=False)


def resolve_trade(row: pd.Series, minutes: pd.DataFrame, cfg: Config, now: pd.Timestamp) -> dict | None:
    """Outcome of one open trade, or None while it is still running."""
    lc = cfg.labels
    start = row["timestamp"]
    H = lc.horizon_h * 60
    m = minutes[(minutes.index >= start) & (minutes.index < start + pd.Timedelta(minutes=H))]
    if m.empty or m.index[0] != start:
        return None                                   # entry minute not available yet
    side = SIDE[row["decision"]]
    entry = float(m["open"].iloc[0])
    tp, sl = entry + side * lc.tp_atr * row["atr"], entry - side * lc.sl_atr * row["atr"]
    label, gross, held = barrier_outcome(m["high"].to_numpy(), m["low"].to_numpy(), float(m["close"].iloc[-1]),
                                         entry, tp, sl, side)
    horizon_done = now >= start + pd.Timedelta(minutes=H) and len(m) == H
    if label == NO_TRADE and not horizon_done:
        return None
    exit_px = entry * (1 + side * gross)
    return {"status": "CLOSED", "entry": entry, "tp": tp, "sl": sl, "exit": exit_px,
            "exit_time": start + pd.Timedelta(minutes=int(held)), "exit_reason": REASON[label],
            "minutes": int(held), "gross_ret": gross, "pnl": float(net_return(side, entry, exit_px, cfg))}


def resolve_open(cfg: Config, log: pd.DataFrame, now: pd.Timestamp, session=None) -> pd.DataFrame:
    log = log.copy()
    for sym, g in log[log["status"] == "OPEN"].groupby("symbol"):
        start = g["timestamp"].min()
        end = min(now.floor("min"), g["timestamp"].max() + pd.Timedelta(hours=cfg.labels.horizon_h))
        minutes = fetch_minutes(sym, cfg, start, end, session=session)
        for i, row in g.iterrows():
            out = resolve_trade(row, minutes, cfg, now)
            if out:
                for k, v in out.items():
                    log.loc[i, k] = v
    return log


def record_signals(cfg: Config, log: pd.DataFrame, kind: str, symbols=None, session=None,
                   now: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    for sym in symbols or cfg.symbols:
        sig, _ = get_signal(cfg, sym, kind, session=session, now=now)
        seen = (log["symbol"] == sym) & (pd.to_datetime(log["bar_time"], utc=True) == sig.bar_time)
        if seen.any():
            continue
        status = "NO TRADE"
        if sig.decision in SIDE:
            is_open = ((log["symbol"] == sym) & (log["status"] == "OPEN")).any()
            status = "SKIPPED" if is_open else "OPEN"
        rows.append({"timestamp": sig.decision_time, "bar_time": sig.bar_time, "symbol": sym, "model": sig.model,
                     "decision": sig.decision, "status": status, "probability": sig.confidence,
                     "p_long_win": sig.p_long_win, "p_short_win": sig.p_short_win, "atr": sig.atr,
                     "entry": sig.entry, "tp": sig.tp, "sl": sig.sl})
        print(f"{sig.decision_time:%Y-%m-%d %H:%M} {sym}: {sig.decision} ({status})")
    if not rows:
        return log
    new = typed(pd.DataFrame(rows))
    return new if log.empty else pd.concat([typed(log), new], ignore_index=True)


def run_once(cfg: Config, kind: str | None = None, session=None, now: pd.Timestamp | None = None) -> pd.DataFrame:
    kind = kind or cfg.live.model_kind
    now = now or pd.Timestamp.now(tz="UTC")
    log = resolve_open(cfg, load_log(cfg), now, session)
    log = record_signals(cfg, log, kind, session=session, now=now)
    save_log(cfg, log)
    return log


def loop(cfg: Config, kind: str | None = None) -> None:
    while True:
        try:
            run_once(cfg, kind)
        except Exception as e:                        # keep running through network hiccups
            print(f"paper trade cycle failed: {e}")
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("h") + pd.Timedelta(hours=1, minutes=cfg.paper.poll_minute)
        time.sleep(max(5.0, (nxt - now).total_seconds()))


def _stats(trades: pd.DataFrame, ret_col: str, t_col: str) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": np.nan, "mean_ret": np.nan, "total_ret": np.nan,
                "tp": 0, "sl": 0, "timeout": 0, "per_week": np.nan}
    r = trades[ret_col].astype(float)
    t = pd.to_datetime(trades[t_col], utc=True)
    weeks = max((t.max() - t.min()).total_seconds() / (7 * 86400), 1)
    reasons = trades["exit_reason"].value_counts()
    return {"trades": len(trades), "win_rate": float((r > 0).mean()), "mean_ret": float(r.mean()),
            "total_ret": float(r.sum()), "tp": int(reasons.get("TP", 0)), "sl": int(reasons.get("SL", 0)),
            "timeout": int(reasons.get("TIMEOUT", 0)), "per_week": len(trades) / weeks}


def summary(cfg: Config, kind: str | None = None) -> str:
    kind = kind or cfg.live.model_kind
    log = load_log(cfg)
    closed = log[log["status"] == "CLOSED"]
    bt_path = resolve_path(cfg, "reports") / f"trades_{kind}.csv"
    bt = pd.read_csv(bt_path) if bt_path.exists() else pd.DataFrame()
    paper = _stats(closed, "pnl", "timestamp")
    back = _stats(bt, "net_ret", "entry_time") if not bt.empty else _stats(pd.DataFrame(), "", "")

    def f(v, pct=True):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.3%}" if pct else f"{v:,.2f}")

    lines = ["# Paper trading vs backtest", "",
             f"Signals logged: {len(log):,} ({(log['status'] == 'NO TRADE').sum():,} no trade, "
             f"{(log['status'] == 'SKIPPED').sum():,} skipped, {(log['status'] == 'OPEN').sum():,} open, "
             f"{len(closed):,} closed). Returns are per trade after fees and slippage.", "",
             "| | Trades | Win rate | Mean trade | Sum of trade returns | TP / SL / timeout | Trades per week |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for name, s in [("Paper (live)", paper), (f"Backtest ({kind} walk-forward)", back)]:
        wr = "n/a" if np.isnan(s["win_rate"]) else f"{s['win_rate']:.1%}"
        lines.append(f"| {name} | {s['trades']:,} | {wr} | {f(s['mean_ret'])} | {f(s['total_ret'])} | "
                     f"{s['tp']} / {s['sl']} / {s['timeout']} | {f(s['per_week'], pct=False)} |")
    if paper["trades"] < 30:
        lines += ["", f"Only {paper['trades']} closed paper trades: far too few to judge the model."]
    md = "\n".join(lines) + "\n"
    path = ensure_dir(resolve_path(cfg, "reports")) / "paper_summary.md"
    path.write_text(md, encoding="utf-8")
    return md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    g.add_argument("--summary", action="store_true")
    p.add_argument("--model")
    a = p.parse_args()
    cfg = load_config()
    if a.summary:
        print(summary(cfg, a.model))
    elif a.loop:
        loop(cfg, a.model)
    else:
        run_once(cfg, a.model)


if __name__ == "__main__":
    main()
