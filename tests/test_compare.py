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
