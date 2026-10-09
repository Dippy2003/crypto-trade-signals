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
