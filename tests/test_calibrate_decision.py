import numpy as np
import pandas as pd
import pytest

from src import calibrate as cb
from src import decision as dc
from src import train as tr
from src.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"decision": {"min_trades": 5}})


def overconfident(n=4000, seed=0):
    """True class probabilities vs a model that pushes them towards the extremes."""
    rng = np.random.default_rng(seed)
    true = rng.dirichlet([2, 1, 1.5], n)
    y = np.array([rng.choice(3, p=p) for p in true])
    raw = true ** 3
    return raw / raw.sum(axis=1, keepdims=True), y


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_improves_log_loss(method):
    raw, y = overconfident()
    cal = cb.Calibrator(method).fit(raw[:3000], y[:3000])
    out = cal.transform(raw[3000:])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    from sklearn.metrics import log_loss
    assert log_loss(y[3000:], out) < log_loss(y[3000:], raw[3000:])


def test_net_return_known_values(cfg):
    # long 100 -> 102 with 0.02% slippage per fill and 0.1% fee per side
    fin, fout = 100 * 1.0002, 102 * 0.9998
    assert dc.net_return(1, np.array([100.0]), np.array([102.0]), cfg)[0] == pytest.approx(
        fout / fin - 1 - 0.001 * (1 + fout / fin))
    # short 100 -> 98 earns about the same as the long above
    assert dc.net_return(-1, np.array([100.0]), np.array([98.0]), cfg)[0] == pytest.approx(0.02 - 0.0024, abs=2e-4)
    assert dc.round_trip_cost(cfg) == pytest.approx(0.0024)


def test_expected_value(cfg):
    p = np.array([[0.5, 0.2, 0.3]])
    # 0.3 * 2 * 1% - 0.5 * 1 * 1% - 0.24% costs
    assert dc.expected_value(p, np.array([0.01]), cfg)[0] == pytest.approx(0.006 - 0.005 - 0.0024)
