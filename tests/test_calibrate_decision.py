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


def test_decide_rule(cfg):
    atr = np.full(5, 0.02)
    pl = np.array([[.3, .1, .6], [.3, .1, .6], [.6, .1, .3], [.3, .1, .6], [.45, .2, .35]])
    ps = np.array([[.6, .1, .3], [.2, .1, .7], [.3, .1, .6], [.3, .1, .6], [.6, .1, .3]])
    dec, ev_l, ev_s = dc.decide(pl, ps, atr, 0.5, 0.5, cfg)
    assert dec[0] == dc.LONG                   # only long passes
    assert dec[1] == dc.SHORT                  # both pass, short has the higher EV
    assert dec[2] == dc.SHORT                  # only short passes
    assert dec[3] == dc.FLAT                   # identical EVs: neither beats the other
    assert dec[4] == dc.FLAT                   # below threshold
    dec_small_atr, *_ = dc.decide(pl[:1], ps[:1], np.array([0.001]), 0.5, 0.5, cfg)
    assert dec_small_atr[0] == dc.FLAT         # EV after costs is negative when ATR is tiny


def test_choose_threshold(cfg):
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 2000)
    net = np.where(p > 0.6, 0.01, -0.01)       # only confident trades make money
    thr, profit, n = dc.choose_threshold(p, np.ones_like(p), net, cfg)
    assert 0.6 <= thr <= 0.62 and profit > 0 and n >= 5
    thr, profit, n = dc.choose_threshold(p, np.ones_like(p), -np.abs(net), cfg)
    assert thr == dc.NO_THRESHOLD and n == 0   # nothing profitable -> do not trade
    thr, *_ = dc.choose_threshold(p, -np.ones_like(p), net, cfg)
    assert thr == dc.NO_THRESHOLD              # negative EV everywhere -> no trades


@pytest.fixture(scope="module")
def cal_oof(project):
    oof = tr.run(project, "logreg")
    return oof, cb.calibrate_oof(oof, project)


def test_first_fold_is_warm_up(project, cal_oof):
    oof, cal = cal_oof
    assert not cal.loc[cal["fold"] == 1, "calibrated"].any()
    assert cal.loc[cal["fold"] > 1, "calibrated"].all()
    c = cal.loc[cal["calibrated"], [f"long_{x}" for x in cb.CAL]].to_numpy()
    np.testing.assert_allclose(c.sum(axis=1), 1.0)


def test_calibration_does_not_see_its_own_fold(project, cal_oof):
    oof, cal = cal_oof
    scrambled = oof.copy()
    k3 = scrambled["fold"] == 3
    scrambled.loc[k3, "long_label"] = np.random.default_rng(0).integers(0, 3, k3.sum())
    again = cb.calibrate_oof(scrambled, project)
    cols = [f"long_{x}" for x in cb.CAL]
    pd.testing.assert_frame_equal(again.loc[k3, cols], cal.loc[k3, cols])
