import numpy as np
import pandas as pd
import pytest

from src import metrics as mt
from src.config import load_config
from src.decision import round_trip_cost

T = pd.Timestamp("2024-01-01", tz="UTC")


@pytest.fixture
def cfg():
    return load_config(overrides={"metrics": {"random_sims": 10}})


def hours(n):
    return pd.date_range(T, periods=n, freq="1h")


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 104], index=hours(5), dtype=float)
    assert mt.max_drawdown(eq) == pytest.approx(90 / 120 - 1)


def test_sharpe_sortino_annualized():
    r = np.array([0.01, -0.005, 0.002, 0.004, -0.001])
    eq = pd.Series(100 * np.cumprod(np.r_[1, 1 + r]), index=hours(6))
    sharpe, sortino = mt.sharpe_sortino(eq, 8760)
    assert sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(8760))
    assert sortino == pytest.approx(r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(8760))
    flat = pd.Series(100.0, index=hours(10))
    assert mt.sharpe_sortino(flat, 8760) == (0.0, 0.0)
