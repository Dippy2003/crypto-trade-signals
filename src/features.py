"""Scale-free features on closed hourly bars.

The row for bar T (open time T, closes at T + 1h) only uses bars with open time <= T, so
it is known at decision time T + 1h. No raw prices: only returns, ratios and oscillators.
Other symbols get the context symbol's (BTC) features as extra ``btc_*`` columns.

Output: data/processed/{SYM}_features.parquet

    python -m src.features
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import Config, load_config, resolve_path
from src.data_loader import load_hourly
from src.labels import wilder_atr

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_features(h: pd.DataFrame, fc: Config) -> pd.DataFrame:
    """Features for one symbol. ``fc`` is the ``features`` config section."""
    # Work on a complete hourly grid so that shift(n) always means n hours; dropped hours are NaN.
    grid = pd.date_range(h.index[0], h.index[-1], freq="1h", name=h.index.name)
    g = h.reindex(grid)
    o, hi, lo, c, v = (g[k] for k in ["open", "high", "low", "close", "volume"])
    f: dict[str, pd.Series] = {}

    for n in fc.return_windows:
        f[f"ret_{n}h"] = c / c.shift(n) - 1

    spans = sorted(fc.ema_spans)
    ema = {s: _ema(c, s) for s in spans}
    for s in spans:
        f[f"close_ema{s}"] = c / ema[s] - 1
    for a, b in zip(spans[:-1], spans[1:]):
        f[f"ema{a}_ema{b}"] = ema[a] / ema[b] - 1

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / fc.rsi_n, adjust=False, min_periods=fc.rsi_n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / fc.rsi_n, adjust=False, min_periods=fc.rsi_n).mean()
    f[f"rsi{fc.rsi_n}"] = up / (up + dn)                     # RSI / 100, defined when up + dn > 0

    macd = (_ema(c, fc.macd_fast) - _ema(c, fc.macd_slow)) / c
    signal = macd.ewm(span=fc.macd_signal, adjust=False, min_periods=fc.macd_signal).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd, signal, macd - signal

    mid = c.rolling(fc.bb_n).mean()
    sd = c.rolling(fc.bb_n).std(ddof=0)
    upper, lower = mid + fc.bb_k * sd, mid - fc.bb_k * sd
    f["bb_pctb"] = (c - lower) / (upper - lower)
    f["bb_width"] = (upper - lower) / mid

    f[f"atr{fc.atr_n}_close"] = wilder_atr(h, fc.atr_n).reindex(grid) / c

    logret = np.log(c).diff()
    for n in fc.vol_windows:
        f[f"rv_{n}h"] = logret.rolling(n).std()

    f["hl_range"] = (hi - lo) / c
    f["co_range"] = c / o - 1
    f[f"vol_rel{fc.volume_mean_n}"] = v / v.rolling(fc.volume_mean_n).mean()

    hour, dow = grid.hour.to_numpy(), grid.dayofweek.to_numpy()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    out = pd.DataFrame({k: np.asarray(s, dtype="float64") for k, s in f.items()}, index=grid)
    return out.reindex(h.index).replace([np.inf, -np.inf], np.nan)
