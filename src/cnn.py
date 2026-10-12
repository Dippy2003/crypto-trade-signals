"""ResNet18 on chart images, trained per symbol, side and walk-forward fold.

Uses the same rows, labels, folds, purging and validation tail as the tabular models (rows
without a complete chart image are dropped). The final layer is replaced with a 3-class head
(LOSS, NO TRADE, WIN); ImageNet weights are used when ``cnn.pretrained`` is true. Training
uses balanced class weights and early stopping on the validation tail, and runs on the GPU
when one is available. Out-of-fold probabilities are saved in the same format as XGBoost,
so calibration, the decision rule and the backtest are reused unchanged.

Output: data/processed/oof_cnn.parquet, models/cnn_{SYM}_{side}_fold{k}.pt, reports/train_cnn.md

    python -m src.cnn [--symbols BTCUSDT] [--folds 5 6] [--epochs 5]
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src import train as tr
from src.chart_images import image_path
from src.config import Config, ensure_dir, load_config, resolve_path
from src.splits import fit_val_masks, fold_masks, make_folds

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TO_TENSOR = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChartDataset(Dataset):
    def __init__(self, paths: list[str], labels: np.ndarray | None = None):
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = TO_TENSOR(Image.open(self.paths[i]).convert("RGB"))
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y


def build_model(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, len(tr.CLASSES))
    return m


def _loader(paths, labels, cfg: Config, shuffle: bool, seed: int = 0) -> DataLoader:
    c = cfg.cnn
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ChartDataset(paths, labels), batch_size=c.batch_size, shuffle=shuffle,
                      num_workers=c.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=c.num_workers > 0, generator=g)


def _loss_fn(y: np.ndarray, dev: torch.device) -> nn.Module:
    counts = np.bincount(y, minlength=len(tr.CLASSES)).astype(np.float32)
    w = len(y) / (len(tr.CLASSES) * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))


def _epoch_loss(model, loader, loss_fn, dev, optimizer=None, scaler=None) -> float:
    model.train(optimizer is not None)
    total, n = 0.0, 0
    amp = dev.type == "cuda"
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(dev.type, enabled=amp):
            loss = loss_fn(model(x), y)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * len(y)
        n += len(y)
    return total / max(n, 1)


def fit_cnn(paths_fit, y_fit, paths_val, y_val, cfg: Config, dev: torch.device | None = None) -> nn.Module:
    """Train with early stopping on validation loss; returns the best model (eval mode)."""
    dev = dev or device()
    c = cfg.cnn
    torch.manual_seed(cfg.model.seed)
    model = build_model(c.pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    fit_loader = _loader(paths_fit, y_fit, cfg, shuffle=True, seed=cfg.model.seed)
    val_loader = _loader(paths_val, y_val, cfg, shuffle=False)
    fit_loss, val_loss = _loss_fn(y_fit, dev), _loss_fn(y_val, dev)
    best, best_state, bad = np.inf, copy.deepcopy(model.state_dict()), 0
    for epoch in range(c.epochs):
        tl = _epoch_loss(model, fit_loader, fit_loss, dev, opt, scaler)
        vl = _epoch_loss(model, val_loader, val_loss, dev)
        print(f"    epoch {epoch + 1}: train {tl:.4f}  val {vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= c.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, paths: list[str], cfg: Config, dev: torch.device | None = None) -> np.ndarray:
    """(n, 3) softmax probabilities in CLASSES order."""
    dev = dev or device()
    out = []
    for x, _ in _loader(paths, None, cfg, shuffle=False):
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out.append(torch.softmax(model(x.to(dev)).float(), dim=1).cpu().numpy())
    p = np.concatenate(out).astype(np.float64) if out else np.zeros((0, 3))
    return p / p.sum(axis=1, keepdims=True)


def image_rows(cfg: Config, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Development rows that have a chart image, and their image paths."""
    ds = tr.development_data(tr.assemble_dataset(cfg, symbol), cfg)
    paths = [image_path(cfg, symbol, t) for t in ds.df.index]
    has = np.array([p.exists() for p in paths])
    return ds.df[has], [str(p) for p, h in zip(paths, has) if h]


def walk_forward_cnn(cfg: Config, symbols: list[str] | None = None,
                     folds: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = device()
    print(f"CNN on {dev}")
    ensure_dir(resolve_path(cfg, "models"))
    rng = np.random.default_rng(cfg.model.seed)
    oof_parts, metrics = [], []
    for sym in symbols or cfg.symbols:
        df, paths = image_rows(cfg, sym)
        if df.empty:
            print(f"{sym}: no chart images; run `python -m src.chart_images` first")
            continue
        t, P = df.index, np.array(paths)
        for fold in make_folds(cfg):
            if folds and fold.k not in folds:
                continue
            train, test = fold_masks(t, fold, cfg)
            if train.sum() < cfg.splits.min_train_rows or not test.any():
                continue
            fit, val = fit_val_masks(t, train, cfg)
            if cfg.cnn.train_subsample < 1:
                fit &= rng.random(len(fit)) < cfg.cnn.train_subsample
            part = pd.DataFrame({"symbol": sym, "fold": fold.k}, index=t[test])
            for side in tr.SIDES:
                y = df[f"{side}_label"].to_numpy()
                print(f"cnn {sym} {side} fold {fold.k}: fit {fit.sum():,} val {val.sum():,} test {test.sum():,}")
                model = fit_cnn(list(P[fit]), y[fit], list(P[val]), y[val], cfg, dev)
                proba = predict(model, list(P[test]), cfg, dev)
                for j, name in enumerate(tr.PROBA):
                    part[f"{side}_{name}"] = proba[:, j]
                part[f"{side}_label"] = y[test]
                prior = np.bincount(y[train], minlength=3) / train.sum()
                metrics.append({"model": "cnn", "symbol": sym, "side": side, "fold": fold.k,
                                "test_start": fold.test_start.date(), "n_train": int(fit.sum()),
                                **tr.fold_metrics(y[test], proba, prior)})
                torch.save(model.state_dict(), resolve_path(cfg, "models") / f"cnn_{sym}_{side}_fold{fold.k}.pt")
            oof_parts.append(part)
    if not oof_parts:
        raise SystemExit("No usable CNN folds (missing images or too little training data)")
    return pd.concat(oof_parts).rename_axis("time").reset_index(), pd.DataFrame(metrics)


def run(cfg: Config, symbols: list[str] | None = None, folds: list[int] | None = None) -> pd.DataFrame:
    oof, m = walk_forward_cnn(cfg, symbols, folds)
    ensure_dir(tr.oof_path(cfg, "cnn").parent)
    oof.to_parquet(tr.oof_path(cfg, "cnn"))
    report = ensure_dir(resolve_path(cfg, "reports")) / "train_cnn.md"
    report.write_text(tr.metrics_markdown(m, "cnn"), encoding="utf-8")
    print(f"Saved {tr.oof_path(cfg, 'cnn')} and {report}")
    return oof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--folds", nargs="+", type=int, help="only these fold numbers (default: all)")
    p.add_argument("--epochs", type=int)
    a = p.parse_args()
    cfg = load_config(overrides={"cnn": {"epochs": a.epochs}} if a.epochs else None)
    run(cfg, a.symbols, a.folds)


if __name__ == "__main__":
    main()
