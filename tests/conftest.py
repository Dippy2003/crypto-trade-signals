"""Shared fixture: a small synthetic project (processed 1m/1h data, labels, features) in a temp folder."""
import pytest

from src import features, labels
from src.config import _merge, ensure_dir, load_config, resolve_path
from src.data_loader import to_hourly
from synth import random_walk_minutes

SMALL = {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "splits": {"first_test_start": "2024-03-01", "test_months": 1, "holdout_start": "2024-06-01",
               "embargo_h": 24, "val_fraction": 0.2, "min_train_rows": 500},
    "model": {"xgb": {"n_estimators": 60, "early_stopping_rounds": 10, "n_jobs": 2}},
    "calibration": {"min_rows": 200},
    "decision": {"min_trades": 5},
    "metrics": {"random_sims": 20, "bootstrap_n": 200},
}


def small_config(root, extra=None):
    paths = {k: str(root / k) for k in ["raw", "processed", "images", "models", "reports"]}
    return load_config(overrides=_merge({**SMALL, "paths": paths}, extra or {}))


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Config pointing at a temp folder with ~5.5 months of synthetic BTC and ETH data processed."""
    root = tmp_path_factory.mktemp("project")
    cfg = small_config(root)
    proc = ensure_dir(resolve_path(cfg, "processed"))
    n = 60 * 24 * 167                                  # 2024-01-01 .. mid June 2024
    for sym, seed, price in [("BTCUSDT", 101, 40000.0), ("ETHUSDT", 102, 2500.0)]:
        m = random_walk_minutes("2024-01-01", n, seed=seed, price=price, vol=0.0012)
        h, _ = to_hourly(m, cfg.data.min_minutes_per_hour)
        m.to_parquet(proc / f"{sym}_1m.parquet")
        h.to_parquet(proc / f"{sym}_1h.parquet")
    labels.build(cfg)
    features.build(cfg)
    return cfg
