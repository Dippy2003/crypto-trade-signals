"""Probability calibration on out-of-fold predictions.

For walk-forward fold k the calibrator is fitted only on OOF predictions from earlier folds
whose label windows closed at least one embargo before fold k starts, so calibrated
probabilities never use information from the period they are applied to. Folds without
enough earlier OOF rows stay uncalibrated and are treated as warm-up (not traded).

Each class (LOSS, NO TRADE, WIN) is calibrated one-vs-rest with isotonic regression or
Platt scaling, then the three probabilities are renormalized to sum to one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.splits import label_end, make_folds
from src.train import CLASSES, PROBA, SIDES

CAL = ["c_loss", "c_none", "c_win"]
EPS = 1e-6


class Calibrator:
    """One-vs-rest calibration of (n, 3) class probabilities."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.maps: list = []

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "Calibrator":
        self.maps = []
        for j, c in enumerate(CLASSES):
            target = (y == c).astype(float)
            if self.method == "isotonic":
                m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(proba[:, j], target)
            else:
                m = LogisticRegression(C=1e6).fit(self._logit(proba[:, j]), target)
            self.maps.append(m)
        return self

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, proba: np.ndarray) -> np.ndarray:
        cols = []
        for j, m in enumerate(self.maps):
            if self.method == "isotonic":
                cols.append(m.predict(proba[:, j]))
            else:
                cols.append(m.predict_proba(self._logit(proba[:, j]))[:, 1])
        out = np.clip(np.column_stack(cols), EPS, None)
        return out / out.sum(axis=1, keepdims=True)


def fit_calibrator(proba: np.ndarray, y: np.ndarray, cfg: Config) -> Calibrator:
    return Calibrator(cfg.calibration.method).fit(proba, y)


def prior_rows(oof: pd.DataFrame, fold_start: pd.Timestamp, k: int, cfg: Config) -> np.ndarray:
    """Rows from earlier folds whose labels were known one embargo before ``fold_start``."""
    emb = pd.Timedelta(hours=cfg.splits.embargo_h)
    ends = label_end(pd.DatetimeIndex(oof["time"]), cfg)
    return np.asarray((oof["fold"] < k) & (ends <= fold_start - emb))


def calibrate_oof(oof: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add ``{side}_c_*`` calibrated columns and a ``calibrated`` flag, walking forward fold by fold."""
    out = oof.copy()
    for side in SIDES:
        for c in CAL:
            out[f"{side}_{c}"] = np.nan
    out["calibrated"] = False
    starts = {f.k: f.test_start for f in make_folds(cfg)}
    for sym, g in oof.groupby("symbol"):
        for k in sorted(g["fold"].unique()):
            prior = g[prior_rows(g, starts[k], k, cfg)]
            if len(prior) < cfg.calibration.min_rows:
                continue
            rows = g.index[g["fold"] == k]
            for side in SIDES:
                cal = fit_calibrator(prior[[f"{side}_{p}" for p in PROBA]].to_numpy(),
                                     prior[f"{side}_label"].to_numpy(), cfg)
                out.loc[rows, [f"{side}_{c}" for c in CAL]] = cal.transform(
                    oof.loc[rows, [f"{side}_{p}" for p in PROBA]].to_numpy())
            out.loc[rows, "calibrated"] = True
    return out


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed frequency per probability bin (for reports)."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "hit": hit})
    g = df.groupby("bin").agg(predicted=("p", "mean"), observed=("hit", "mean"), n=("p", "size"))
    return g.reset_index(drop=True)
