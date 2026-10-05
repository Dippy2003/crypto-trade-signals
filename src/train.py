"""Dataset assembly and walk-forward training of per-side classifiers.

One model per symbol, side (long / short) and fold predicts P(LOSS), P(NO TRADE), P(WIN).
Test-period predictions of every fold are out-of-fold (OOF) and are saved for calibration,
the decision rule and the backtest.

Output: data/processed/oof_{model}.parquet, reports/train_{model}.md

    python -m src.train --model logreg
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config, ensure_dir, load_config, resolve_path
from src.data_loader import load_hourly
from src.features import load_features
from src.labels import LABEL_NAMES, LOSS, NO_TRADE, WIN, load_labels
from src.splits import assert_no_holdout, development_mask, fit_val_masks, fold_masks, make_folds

SIDES = ("long", "short")
CLASSES = [LOSS, NO_TRADE, WIN]
PROBA = ["p_loss", "p_none", "p_win"]


@dataclass
class Dataset:
    df: pd.DataFrame          # features + labels + close, indexed by bar time
    features: list[str]
    symbol: str


def assemble_dataset(cfg: Config, symbol: str) -> Dataset:
    """Join features, labels and the bar close on time; drop rows with missing features."""
    feats = load_features(cfg, symbol)
    df = feats.join(load_labels(cfg, symbol), how="inner").join(load_hourly(cfg, symbol)[["close"]])
    df = df.dropna(subset=list(feats.columns))
    return Dataset(df, list(feats.columns), symbol)


def development_data(ds: Dataset, cfg: Config) -> Dataset:
    """The same dataset without any row that touches the holdout."""
    df = ds.df[development_mask(ds.df.index, cfg)]
    assert_no_holdout(df.index, cfg)
    return Dataset(df, ds.features, ds.symbol)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities in CLASSES order, even if a class was absent in training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(int(c))] = p[:, j]
    return out


def fit_logreg(X_fit, y_fit, X_val, y_val, cfg: Config):
    """Standardized logistic regression with balanced class weights. Scaler sees training rows only."""
    X = np.vstack([X_fit, X_val])
    y = np.concatenate([y_fit, y_val])
    lr = cfg.model.logreg
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=lr.C, max_iter=lr.max_iter, class_weight="balanced", random_state=cfg.model.seed))
    return model.fit(X, y)


TRAINERS: dict[str, Callable] = {"logreg": fit_logreg}


def fold_metrics(y: np.ndarray, proba: np.ndarray, prior: np.ndarray) -> dict:
    """Log-loss (model and class-prior baseline), per-class precision/recall, confusion matrix."""
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    prec, rec, _, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {
        "log_loss": log_loss(y, np.clip(proba, 1e-12, 1), labels=CLASSES),
        "prior_log_loss": log_loss(y, np.tile(prior, (len(y), 1)), labels=CLASSES),
        **{f"precision_{LABEL_NAMES[c]}": p for c, p in zip(CLASSES, prec)},
        **{f"recall_{LABEL_NAMES[c]}": r for c, r in zip(CLASSES, rec)},
        "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "n_test": len(y),
    }
