"""
Step 2: Triple-barrier labels (WIN / NO TRADE / LOSS) for LONG and SHORT.

For each hourly bar:
  - Decision time = when the bar closes (e.g. bar 10:00 closes at 11:00)
  - Entry        = open of the next minute (11:00)
  - LONG : TP = entry + TP_ATR*ATR,  SL = entry - SL_ATR*ATR
  - SHORT: TP = entry - TP_ATR*ATR,  SL = entry + SL_ATR*ATR
  - Walk forward minute by minute for HORIZON_H hours:
        TP first      -> 2 (WIN)
        SL first      -> 0 (LOSS)   (both in same minute -> LOSS, conservative)
        neither       -> 1 (NO TRADE / timeout)

Input : data/processed/{SYM}_1m.parquet, {SYM}_1h.parquet
Output: data/processed/{SYM}_labels.parquet
"""
import glob
import os

import numpy as np
import pandas as pd

# ---- settings (tune later on validation data, never on test) ----
TP_ATR = 2.0
SL_ATR = 1.0
HORIZON_H = 12
ATR_N = 14
PROC = "data/processed"

LOSS, NO_TRADE, WIN = 0, 1, 2
H = HORIZON_H * 60
ONE_MIN = np.timedelta64(1, "m")


def first_hit(mask):
    i = int(np.argmax(mask))
    return i if mask[i] else np.inf


def barrier(hi, lo, last_close, entry, tp, sl, side):
    """side = +1 long, -1 short. Returns (label, return, minutes_held)."""
    if side == 1:
        t_tp, t_sl = first_hit(hi >= tp), first_hit(lo <= sl)
    else:
        t_tp, t_sl = first_hit(lo <= tp), first_hit(hi >= sl)

    if t_tp == np.inf and t_sl == np.inf:
        return NO_TRADE, side * (last_close / entry - 1), len(hi)
    if t_sl <= t_tp:
        return LOSS, side * (sl / entry - 1), t_sl + 1
    return WIN, side * (tp / entry - 1), t_tp + 1


for path in sorted(glob.glob(f"{PROC}/*_1h.parquet")):
    sym = os.path.basename(path).split("_")[0]
    h = pd.read_parquet(path)
    m = pd.read_parquet(f"{PROC}/{sym}_1m.parquet")
    print(f"\n{sym}: labelling {len(h):,} hourly bars...")

    # ATR (Wilder) on closed hourly bars
    pc = h["close"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(), (h["low"] - pc).abs()], axis=1).max(axis=1)
    h["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()

    mt = m.index.values
    mo, mh, ml, mc = (m[c].to_numpy() for c in ["open", "high", "low", "close"])
    decision = (h.index + pd.Timedelta(hours=1)).values
    start = np.searchsorted(mt, decision)
    atr = h["atr"].to_numpy()

    rows = []
    for i in range(len(h)):
        s, a = start[i], atr[i]
        e = s + H
        # skip: no ATR yet, not enough future data, entry minute missing, or gaps in window
        if np.isnan(a) or e > len(mt) or mt[s] != decision[i] or mt[e - 1] - mt[s] != (H - 1) * ONE_MIN:
            continue
        entry = mo[s]
        hi, lo = mh[s:e], ml[s:e]
        L = barrier(hi, lo, mc[e - 1], entry, entry + TP_ATR * a, entry - SL_ATR * a, +1)
        S = barrier(hi, lo, mc[e - 1], entry, entry - TP_ATR * a, entry + SL_ATR * a, -1)
        rows.append((h.index[i], entry, a, *L, *S))

    lab = pd.DataFrame(rows, columns=[
        "time", "entry", "atr",
        "long_label", "long_ret", "long_minutes",
        "short_label", "short_ret", "short_minutes"]).set_index("time")
    lab.to_parquet(f"{PROC}/{sym}_labels.parquet")

    names = {LOSS: "LOSS", NO_TRADE: "NO TRADE", WIN: "WIN"}
    print(f"  labelled: {len(lab):,} rows (skipped {len(h) - len(lab):,})")
    for side in ["long", "short"]:
        dist = lab[f"{side}_label"].map(names).value_counts(normalize=True).mul(100).round(1)
        print(f"  {side:5s}: " + "  ".join(f"{k} {v}%" for k, v in dist.items()))
    print(f"  saved   : {PROC}/{sym}_labels.parquet")
