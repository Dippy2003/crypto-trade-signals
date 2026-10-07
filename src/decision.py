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
