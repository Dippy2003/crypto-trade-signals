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


def walk_forward(cfg: Config, kind: str, symbols: list[str] | None = None,
                 on_model: Callable | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train ``kind`` on every usable fold. Returns (oof predictions, per-fold metrics).

    ``on_model(kind, symbol, side, fold, model, dataset, masks)`` is called after each fit.
    """
    fit_fn = TRAINERS[kind]
    oof_parts, metrics = [], []
    for sym in symbols or cfg.symbols:
        ds = development_data(assemble_dataset(cfg, sym), cfg)
        X, t = ds.df[ds.features].to_numpy(), ds.df.index
        for fold in make_folds(cfg):
            train, test = fold_masks(t, fold, cfg)
            if train.sum() < cfg.splits.min_train_rows or not test.any():
                continue
            fit, val = fit_val_masks(t, train, cfg)
            part = pd.DataFrame({"symbol": sym, "fold": fold.k}, index=t[test])
            for side in SIDES:
                y = ds.df[f"{side}_label"].to_numpy()
                model = fit_fn(X[fit], y[fit], X[val], y[val], cfg)
                proba = predict_proba(model, X[test])
                for j, name in enumerate(PROBA):
                    part[f"{side}_{name}"] = proba[:, j]
                part[f"{side}_label"] = y[test]
                prior = np.bincount(y[train], minlength=3) / train.sum()
                metrics.append({"model": kind, "symbol": sym, "side": side, "fold": fold.k,
                                "test_start": fold.test_start.date(), "n_train": int(train.sum()),
                                **fold_metrics(y[test], proba, prior)})
                if on_model:
                    on_model(kind, sym, side, fold, model, ds, (fit, val, test))
            oof_parts.append(part)
            print(f"{kind} {sym} fold {fold.k}: train {train.sum():,}  test {test.sum():,}")
    if not oof_parts:
        raise SystemExit("No usable folds: check data range and splits.first_test_start")
    oof = pd.concat(oof_parts).rename_axis("time").reset_index()
    return oof, pd.DataFrame(metrics)


def metrics_markdown(m: pd.DataFrame, kind: str) -> str:
    lines = [f"# Training report: {kind}", "",
             "Out-of-fold metrics per test period. `prior` is the log-loss of always predicting the "
             "training class frequencies; a useful model should be below it.", ""]
    for (sym, side), g in m.groupby(["symbol", "side"], sort=False):
        lines += [f"## {sym} {side}", "",
                  "| Fold | Test start | Train | Test | Log-loss | Prior | Prec WIN | Rec WIN | Prec LOSS | Rec LOSS | Confusion [L,N,W] |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in g.to_dict("records"):
            lines.append(f"| {r['fold']} | {r['test_start']} | {r['n_train']:,} | {r['n_test']:,} | "
                         f"{r['log_loss']:.4f} | {r['prior_log_loss']:.4f} | {r['precision_WIN']:.3f} | "
                         f"{r['recall_WIN']:.3f} | {r['precision_LOSS']:.3f} | {r['recall_LOSS']:.3f} | "
                         f"{r['confusion']} |")
        lines += ["", f"Mean log-loss {g['log_loss'].mean():.4f} vs prior {g['prior_log_loss'].mean():.4f}", ""]
    return "\n".join(lines)


def oof_path(cfg: Config, kind: str):
    return resolve_path(cfg, "processed") / f"oof_{kind}.parquet"


def load_oof(cfg: Config, kind: str) -> pd.DataFrame:
    return pd.read_parquet(oof_path(cfg, kind))


def run(cfg: Config, kind: str, symbols: list[str] | None = None) -> pd.DataFrame:
    oof, m = walk_forward(cfg, kind, symbols)
    ensure_dir(oof_path(cfg, kind).parent)
    oof.to_parquet(oof_path(cfg, kind))
    report = ensure_dir(resolve_path(cfg, "reports")) / f"train_{kind}.md"
    report.write_text(metrics_markdown(m, kind), encoding="utf-8")
    print(f"Saved {oof_path(cfg, kind)} and {report}")
    return oof
