"""Compare XGBoost (numbers), CNN (chart images), logistic regression and random entries.

All models are evaluated on the same (symbol, time) rows, the same folds, the same
calibration warm-up, the same decision rule, costs and backtester. Significance: a bootstrap
confidence interval on the difference in mean net trade return between each pair.

Output: reports/comparison.md

    python -m src.compare [--models xgb cnn logreg]
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from src.backtest import backtest
from src.calibrate import calibrate_oof
from src.config import Config, ensure_dir, load_config, resolve_path
from src.decision import LONG, SHORT, attach_market, decide_oof
from src.evaluate import H1, setup_lines
from src.metrics import metrics_table, plot_equity, summarize
from src.train import load_oof, oof_path

DEFAULT_MODELS = ["xgb", "cnn", "logreg"]
KEY = ["symbol", "time"]


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, level: float, seed: int) -> tuple[float, float, float]:
    """Difference in means (a - b) with a percentile bootstrap CI, resampling each side independently."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)
