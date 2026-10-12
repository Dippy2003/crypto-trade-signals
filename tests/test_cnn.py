import copy

import numpy as np
import pytest
import torch

from src import chart_images as ci
from src import cnn
from src import evaluate as ev
from src import train as tr


@pytest.fixture(scope="module")
def cnn_cfg(project, tmp_path_factory):
    """Tiny setup: 64px images for every 12th BTC row, no pretrained download, CPU-friendly."""
    root = tmp_path_factory.mktemp("cnn")
    cfg = copy.deepcopy(project)
    cfg.paths.images, cfg.paths.models, cfg.paths.reports = (str(root / k) for k in ("images", "models", "reports"))
    cfg.images.size_px, cfg.images.workers = 64, 1
    cfg.cnn.update({"pretrained": False, "epochs": 2, "patience": 1, "batch_size": 32, "num_workers": 0})
    cfg.splits.min_train_rows = 50
    cfg.calibration.min_rows = 20
    cfg.symbols = ["BTCUSDT"]
    times = tr.assemble_dataset(cfg, "BTCUSDT").df.index[::12]
    ci.render_symbol(cfg, "BTCUSDT", times, workers=1)
    return cfg


def test_model_head_and_device():
    m = cnn.build_model(pretrained=False)
    assert m.fc.out_features == 3
    assert cnn.device().type in ("cpu", "cuda")
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 3)


def test_dataset_normalizes_images(cnn_cfg):
    df, paths = cnn.image_rows(cnn_cfg, "BTCUSDT")
    x, y = cnn.ChartDataset(paths[:2], df["long_label"].to_numpy()[:2])[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert x.min() < 0 < x.max()                              # ImageNet normalization applied
    assert y in (0, 1, 2)
