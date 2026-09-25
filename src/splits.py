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


def label_end(times: pd.DatetimeIndex, cfg: Config) -> pd.DatetimeIndex:
    """Time at which the label of each row is fully known."""
    return times + pd.Timedelta(hours=1 + cfg.labels.horizon_h)


def outside(times: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, cfg: Config) -> np.ndarray:
    """True for rows whose [T, label end] interval stays ``embargo_h`` away from [start, end)."""
    emb = pd.Timedelta(hours=cfg.splits.embargo_h)
    ends = label_end(times, cfg)
    return np.asarray((ends <= start - emb) | (times >= end + emb))


def make_folds(cfg: Config) -> list[Fold]:
    sp = cfg.splits
    start, stop = _ts(sp.first_test_start), _ts(sp.holdout_start)
    folds, k = [], 1
    while start < stop:
        end = min(start + pd.DateOffset(months=sp.test_months), stop)
        folds.append(Fold(k, start, end))
        start, k = end, k + 1
    return folds


def development_mask(times: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """Rows that never touch the holdout period (usable for training, tuning and walk-forward)."""
    emb = pd.Timedelta(hours=cfg.splits.embargo_h)
    return np.asarray(label_end(times, cfg) <= _ts(cfg.splits.holdout_start) - emb)


def holdout_mask(times: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """Rows in the final holdout. Only the one-shot holdout evaluation may use this."""
    return np.asarray(times >= _ts(cfg.splits.holdout_start))


def fold_masks(times: pd.DatetimeIndex, fold: Fold, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) boolean masks for one fold. Train is purged and embargoed; neither touches the holdout."""
    dev = development_mask(times, cfg)
    test = dev & np.asarray((times >= fold.test_start) & (times < fold.test_end))
    train = dev & np.asarray(times < fold.test_start) & outside(times, fold.test_start, fold.test_end, cfg)
    return train, test


def assert_no_holdout(times: pd.DatetimeIndex, cfg: Config) -> None:
    if not development_mask(times, cfg).all():
        raise RuntimeError("Holdout data reached development code; only `evaluate --holdout` may use it.")
