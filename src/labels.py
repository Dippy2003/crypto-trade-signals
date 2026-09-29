"""Triple-barrier labels (WIN / NO TRADE / LOSS) for LONG and SHORT.

For each hourly bar T:
  - Decision time = when the bar closes (bar 10:00 closes at 11:00)
  - Entry        = open of the next minute (11:00)
  - LONG : TP = entry + tp_atr*ATR,  SL = entry - sl_atr*ATR
  - SHORT: TP = entry - tp_atr*ATR,  SL = entry + sl_atr*ATR
  - Walk forward minute by minute for horizon_h hours:
        TP first      -> 2 (WIN)
        SL first      -> 0 (LOSS)   (both in the same minute -> LOSS, conservative)
        neither       -> 1 (NO TRADE / timeout, exit at the last close)
Rows are skipped when ATR is not ready, the entry minute is missing, the window has a gap,
or there is not enough future data.

Input : data/processed/{SYM}_1m.parquet, {SYM}_1h.parquet
Output: data/processed/{SYM}_labels.parquet, reports/label_distribution.md

    python -m src.labels
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import Config, ensure_dir, load_config, resolve_path
from src.data_loader import load_hourly, load_minute

LOSS, NO_TRADE, WIN = 0, 1, 2
LABEL_NAMES = {LOSS: "LOSS", NO_TRADE: "NO TRADE", WIN: "WIN"}
ONE_MIN = np.timedelta64(1, "m")
LABEL_COLS = ["entry", "atr",
              "long_label", "long_ret", "long_minutes", "long_exit",
              "short_label", "short_ret", "short_minutes", "short_exit"]


def wilder_atr(h: pd.DataFrame, n: int) -> pd.Series:
    """ATR with Wilder smoothing on closed hourly bars."""
    pc = h["close"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(), (h["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def barrier_outcome(hi: np.ndarray, lo: np.ndarray, last_close: float, entry: float,
                    tp: float, sl: float, side: int) -> tuple[int, float, int]:
    """Single-trade version. side = +1 long, -1 short. Returns (label, return, minutes_held)."""
    def first_hit(mask: np.ndarray) -> float:
        i = int(np.argmax(mask)) if len(mask) else 0
        return i if len(mask) and mask[i] else np.inf

    if side == 1:
        t_tp, t_sl = first_hit(hi >= tp), first_hit(lo <= sl)
    else:
        t_tp, t_sl = first_hit(lo <= tp), first_hit(hi >= sl)
    if t_tp == np.inf and t_sl == np.inf:
        return NO_TRADE, side * (last_close / entry - 1), len(hi)
    if t_sl <= t_tp:
        return LOSS, side * (sl / entry - 1), int(t_sl) + 1
    return WIN, side * (tp / entry - 1), int(t_tp) + 1
