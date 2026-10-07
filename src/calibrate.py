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
