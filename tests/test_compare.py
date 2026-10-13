import numpy as np
import pandas as pd

from src import compare as cp
from src import train as tr
from src.config import resolve_path


def test_bootstrap_diff_detects_real_difference():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.01, 0.02, 400), rng.normal(0.0, 0.02, 400)
    d, lo, hi = cp.bootstrap_diff(a, b, 2000, 0.95, 1)
    assert lo < d < hi and lo > 0                         # clear difference: CI excludes zero
    d, lo, hi = cp.bootstrap_diff(a, a + rng.normal(0, 1e-6, 400), 2000, 0.95, 1)
    assert lo < 0 < hi                                    # no difference: CI contains zero


def test_common_rows_intersection():
    t = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    a = pd.DataFrame({"symbol": "BTCUSDT", "time": t, "fold": 1})
    b = pd.DataFrame({"symbol": "BTCUSDT", "time": t[1:], "fold": 1})
    out = cp.common_rows({"a": a, "b": b})
    assert len(out["a"]) == len(out["b"]) == 4
    assert (out["a"]["time"].to_numpy() == out["b"]["time"].to_numpy()).all()
