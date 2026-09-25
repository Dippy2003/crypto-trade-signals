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
