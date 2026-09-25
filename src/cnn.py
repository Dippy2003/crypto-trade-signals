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


def build_model(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, len(tr.CLASSES))
    return m
