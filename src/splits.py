"""Walk-forward folds with purging and an embargo, plus the final holdout.

A row for bar T uses information from T (features, known at T + 1h) up to the end of its
label window, T + 1h + horizon. A training row is *purged* when that interval comes within
``embargo_h`` of a test period. Test periods are consecutive calendar blocks
(``test_months`` long) from ``first_test_start`` up to ``holdout_start``; each fold trains on
everything before its test period (expanding window).

Rows touching the holdout (T + 1h + horizon later than holdout_start - embargo) are excluded
from all development work. Only ``evaluate --holdout`` calls ``holdout_mask``.

    python -m src.splits            # print the fold table for each symbol
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, load_config


@dataclass(frozen=True)
class Fold:
    k: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp            # exclusive


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_folds(cfg: Config) -> list[Fold]:
    sp = cfg.splits
    start, stop = _ts(sp.first_test_start), _ts(sp.holdout_start)
    folds, k = [], 1
    while start < stop:
        end = min(start + pd.DateOffset(months=sp.test_months), stop)
        folds.append(Fold(k, start, end))
        start, k = end, k + 1
    return folds
