"""Dataset assembly and walk-forward training of per-side classifiers.

One model per symbol, side (long / short) and fold predicts P(LOSS), P(NO TRADE), P(WIN).
Test-period predictions of every fold are out-of-fold (OOF) and are saved for calibration,
the decision rule and the backtest.

Output: data/processed/oof_{model}.parquet, models/{model}_{SYM}_{side}_fold{k}.pkl,
        reports/train_{model}.md, reports/figures/importance_{model}_{SYM}_{side}.png

    python -m src.train --model logreg
    python -m src.train --model xgb
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights: n / (n_classes * count[class])."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = len(y) / (len(CLASSES) * np.maximum(counts, 1))
    return w[y]


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities in CLASSES order, even if a class was absent in training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(int(c))] = p[:, j]
    return out / out.sum(axis=1, keepdims=True)          # float32 models do not sum to exactly 1


def fit_logreg(X_fit, y_fit, X_val, y_val, cfg: Config):
    """Standardized logistic regression with balanced class weights. Scaler sees training rows only."""
    X = np.vstack([X_fit, X_val])
    y = np.concatenate([y_fit, y_val])
    lr = cfg.model.logreg
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=lr.C, max_iter=lr.max_iter, class_weight="balanced", random_state=cfg.model.seed))
    return model.fit(X, y)


def fit_xgb(X_fit, y_fit, X_val, y_val, cfg: Config):
    """Gradient-boosted trees with balanced weights and early stopping on the purged validation tail."""
    xc = cfg.model.xgb
    model = XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
        n_estimators=xc.n_estimators, learning_rate=xc.learning_rate, max_depth=xc.max_depth,
        min_child_weight=xc.min_child_weight, subsample=xc.subsample, colsample_bytree=xc.colsample_bytree,
        reg_lambda=xc.reg_lambda, early_stopping_rounds=xc.early_stopping_rounds,
        n_jobs=xc.n_jobs, random_state=cfg.model.seed)
    model.fit(X_fit, y_fit, sample_weight=class_weights(y_fit),
              eval_set=[(X_val, y_val)], sample_weight_eval_set=[class_weights(y_val)], verbose=False)
    return model


TRAINERS: dict[str, Callable] = {"logreg": fit_logreg, "xgb": fit_xgb}


def importance(model, features: list[str]) -> pd.Series:
    """Gain importance for XGBoost, mean |standardized coefficient| for logistic regression."""
    if isinstance(model, XGBClassifier):
        vals = model.feature_importances_
    else:
        vals = np.abs(model[-1].coef_).mean(axis=0)
    s = pd.Series(vals, index=features, dtype=float)
    return s / s.sum() if s.sum() > 0 else s


def plot_importance(imp: pd.Series, title: str, path, top_n: int) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = imp.sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(top) + 1.2))
    ax.barh(top.index, top.values, color="#4c72b0")
    ax.set_xlabel("share of importance (mean over folds)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def model_path(cfg: Config, kind: str, symbol: str, side: str, fold: int):
    return resolve_path(cfg, "models") / f"{kind}_{symbol}_{side}_fold{fold}.pkl"


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
    ensure_dir(resolve_path(cfg, "models"))
    imps: dict[tuple[str, str], list[pd.Series]] = {}

    def on_model(kind, sym, side, fold, model, ds, masks):
        joblib.dump({"model": model, "features": ds.features}, model_path(cfg, kind, sym, side, fold.k))
        imps.setdefault((sym, side), []).append(importance(model, ds.features))

    oof, m = walk_forward(cfg, kind, symbols, on_model=on_model)
    ensure_dir(oof_path(cfg, kind).parent)
    oof.to_parquet(oof_path(cfg, kind))

    reports = ensure_dir(resolve_path(cfg, "reports"))
    figs = ensure_dir(reports / "figures")
    top_n = cfg.model.importance_top_n
    lines = [metrics_markdown(m, kind), "## Feature importance", ""]
    for (sym, side), parts in imps.items():
        imp = pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)
        fig = figs / f"importance_{kind}_{sym}_{side}.png"
        plot_importance(imp, f"{kind} {sym} {side}", fig, top_n)
        lines += [f"### {sym} {side}", "", f"![importance](figures/{fig.name})", "",
                  "Top 5: " + ", ".join(f"{k} ({v:.3f})" for k, v in imp.head(5).items()), ""]
    report = reports / f"train_{kind}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {oof_path(cfg, kind)} and {report}")
    return oof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="logreg", choices=sorted(TRAINERS))
    p.add_argument("--symbols", nargs="+")
    a = p.parse_args()
    run(load_config(), a.model, a.symbols)


if __name__ == "__main__":
    main()
