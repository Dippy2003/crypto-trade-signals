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


def _first_true(mask: np.ndarray, none: int) -> np.ndarray:
    """Index of the first True per row, or ``none`` when a row has no True."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), none)


def _resolve(t_tp, t_sl, entry, tp, sl, last_close, side, H):
    no_hit = (t_tp == H) & (t_sl == H)
    loss = ~no_hit & (t_sl <= t_tp)
    label = np.where(no_hit, NO_TRADE, np.where(loss, LOSS, WIN))
    exit_px = np.where(no_hit, last_close, np.where(loss, sl, tp))
    ret = side * (exit_px / entry - 1)
    minutes = np.where(no_hit, H, np.where(loss, t_sl + 1, t_tp + 1))
    return label, ret, minutes, exit_px


def label_symbol(h: pd.DataFrame, m: pd.DataFrame, tp_atr: float, sl_atr: float,
                 horizon_h: int, atr_n: int, chunk_rows: int = 2048) -> pd.DataFrame:
    """Vectorized triple-barrier labels for one symbol, indexed by hourly bar open time."""
    H = horizon_h * 60
    atr = wilder_atr(h, atr_n).to_numpy()
    mt = m.index.values
    mo, mh, ml, mc = (m[c].to_numpy() for c in ["open", "high", "low", "close"])
    decision = (h.index + pd.Timedelta(hours=1)).values
    start = np.searchsorted(mt, decision)

    rows = np.flatnonzero(~np.isnan(atr) & (start + H <= len(mt)))
    s = start[rows]
    ok = (mt[s] == decision[rows]) & (mt[s + H - 1] - mt[s] == (H - 1) * ONE_MIN)
    rows, s = rows[ok], s[ok]

    entry, a = mo[s], atr[rows]
    last_close = mc[s + H - 1]
    tp_l, sl_l = entry + tp_atr * a, entry - sl_atr * a
    tp_s, sl_s = entry - tp_atr * a, entry + sl_atr * a

    t = np.empty((4, len(rows)), dtype=np.int64)
    offsets = np.arange(H)
    for c0 in range(0, len(rows), chunk_rows):
        c = slice(c0, c0 + chunk_rows)
        win = s[c, None] + offsets
        hi, lo = mh[win], ml[win]
        t[0, c] = _first_true(hi >= tp_l[c, None], H)
        t[1, c] = _first_true(lo <= sl_l[c, None], H)
        t[2, c] = _first_true(lo <= tp_s[c, None], H)
        t[3, c] = _first_true(hi >= sl_s[c, None], H)

    L = _resolve(t[0], t[1], entry, tp_l, sl_l, last_close, +1, H)
    S = _resolve(t[2], t[3], entry, tp_s, sl_s, last_close, -1, H)
    lab = pd.DataFrame({
        "entry": entry, "atr": a,
        "long_label": L[0], "long_ret": L[1], "long_minutes": L[2], "long_exit": L[3],
        "short_label": S[0], "short_ret": S[1], "short_minutes": S[2], "short_exit": S[3],
    }, index=h.index[rows])
    lab.index.name = "time"
    return lab
