"""Cost-aware decision rule: LONG, SHORT or NO TRADE.

LONG  if P(long WIN)  >= long threshold,  EV(long)  > 0 after costs and EV(long)  beats SHORT
SHORT if P(short WIN) >= short threshold, EV(short) > 0 after costs and EV(short) beats LONG
otherwise NO TRADE.

EV (fraction of entry) = P(WIN)*tp_atr*ATR/price - P(LOSS)*sl_atr*ATR/price
                         + P(NO TRADE)*timeout_return - round-trip costs.

Thresholds are chosen per symbol and side on earlier (validation) folds only, by the realized
net profit of the trades they would have taken; if no threshold is profitable the side is not traded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibrate import CAL, prior_rows
from src.config import Config
from src.data_loader import load_hourly
from src.labels import load_labels
from src.splits import make_folds
from src.train import SIDES

LONG, FLAT, SHORT = 1, 0, -1
NAMES = {LONG: "LONG", FLAT: "NO TRADE", SHORT: "SHORT"}
NO_THRESHOLD = np.inf


def round_trip_cost(cfg: Config) -> float:
    c = cfg.costs
    return 2 * (c.fee_per_side + c.slippage_per_side)


def net_return(side: np.ndarray | int, entry: np.ndarray, exit_: np.ndarray, cfg: Config) -> np.ndarray:
    """Return after slippage on both fills and a fee on both notionals."""
    c = cfg.costs
    fill_in = entry * (1 + side * c.slippage_per_side)
    fill_out = exit_ * (1 - side * c.slippage_per_side)
    return side * (fill_out / fill_in - 1) - c.fee_per_side * (1 + fill_out / fill_in)


def expected_value(p: np.ndarray, atr_frac: np.ndarray, cfg: Config) -> np.ndarray:
    """EV per unit notional for (n, 3) calibrated probabilities [LOSS, NO TRADE, WIN]."""
    lc = cfg.labels
    return (p[:, 2] * lc.tp_atr * atr_frac - p[:, 0] * lc.sl_atr * atr_frac
            + p[:, 1] * cfg.decision.timeout_return - round_trip_cost(cfg))


def decide(p_long: np.ndarray, p_short: np.ndarray, atr_frac: np.ndarray,
           thr_long: float, thr_short: float, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (decision in {1, 0, -1}, EV long, EV short)."""
    ev_l, ev_s = expected_value(p_long, atr_frac, cfg), expected_value(p_short, atr_frac, cfg)
    ok_l = (p_long[:, 2] >= thr_long) & (ev_l > 0)
    ok_s = (p_short[:, 2] >= thr_short) & (ev_s > 0)
    go_l = ok_l & (~ok_s | (ev_l > ev_s))
    go_s = ok_s & (~ok_l | (ev_s > ev_l))
    return np.where(go_l, LONG, np.where(go_s, SHORT, FLAT)), ev_l, ev_s


def threshold_grid(cfg: Config) -> np.ndarray:
    d = cfg.decision
    return np.round(np.arange(d.threshold_min, d.threshold_max + d.threshold_step / 2, d.threshold_step), 6)


def choose_threshold(p_win: np.ndarray, ev: np.ndarray, net: np.ndarray, cfg: Config) -> tuple[float, float, int]:
    """Threshold with the highest total net profit (>= min_trades trades). Returns (thr, profit, trades).

    If no threshold makes money the side is switched off (threshold = inf).
    """
    best = (NO_THRESHOLD, 0.0, 0)
    for thr in threshold_grid(cfg):
        sel = (p_win >= thr) & (ev > 0)
        n = int(sel.sum())
        if n < cfg.decision.min_trades:
            continue
        profit = float(net[sel].sum())
        if profit > best[1]:
            best = (float(thr), profit, n)
    return best


def attach_market(oof: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Join bar close and minute-accurate outcomes (entry, exit, minutes, returns) onto OOF rows."""
    parts = []
    for sym, g in oof.groupby("symbol", sort=False):
        lab = load_labels(cfg, sym).drop(columns=[f"{s}_label" for s in SIDES])
        mk = lab.join(load_hourly(cfg, sym)[["close"]])
        parts.append(g.join(mk, on="time"))
    out = pd.concat(parts).sort_index()
    out["atr_frac"] = out["atr"] / out["close"]
    for side, sgn in (("long", LONG), ("short", SHORT)):
        out[f"{side}_net"] = net_return(sgn, out["entry"].to_numpy(), out[f"{side}_exit"].to_numpy(), cfg)
    return out
