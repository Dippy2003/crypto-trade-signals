"""Leakage tests: features at bar T must not change when later data is removed."""
import numpy as np
import pandas as pd
import pytest

from src import features as ft
from src.config import load_config
from src.data_loader import to_hourly
from src.labels import label_symbol
from synth import random_walk_minutes


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data():
    mb = random_walk_minutes("2024-01-01", 60 * 24 * 30, seed=31, price=40000, vol=0.0015)
    me = random_walk_minutes("2024-01-01", 60 * 24 * 30, seed=32, price=2500, vol=0.002)
    mb = mb.drop(mb.index[20000:20070])            # a dropped hour on BTC only
    hb, _ = to_hourly(mb, 50)
    he, _ = to_hourly(me, 50)
    return {"BTCUSDT": hb, "ETHUSDT": he}, {"BTCUSDT": mb, "ETHUSDT": me}


def random_times(h: pd.DataFrame, n: int, seed: int) -> list[pd.Timestamp]:
    rng = np.random.default_rng(seed)
    return list(h.index[np.sort(rng.choice(np.arange(250, len(h) - 1), n, replace=False))])


def truncation_mismatches(build, hourly: dict, times: list[pd.Timestamp]) -> list[str]:
    """Rebuild features on data with timestamps < T + 1h and compare row T with the full build."""
    full = build(hourly)
    bad = []
    for t in times:
        cut = {s: h[h.index < t + pd.Timedelta(hours=1)] for s, h in hourly.items()}
        part = build(cut)
        for sym in full:
            if t not in full[sym].index:
                continue
            a, b = full[sym].loc[t], part[sym].loc[t]
            same = np.isclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=0, equal_nan=True)
            if not same.all():
                bad.append(f"{sym} {t}: {list(a.index[~same])}")
    return bad


def test_features_identical_after_truncation(cfg, data):
    hourly, _ = data
    btc_idx = hourly["BTCUSDT"].index
    after_gap = list(btc_idx[1:][np.diff(btc_idx) > pd.Timedelta(hours=1)])
    assert after_gap                                  # the dropped BTC hours are in the sample
    times = random_times(hourly["ETHUSDT"], 25, seed=1) + after_gap
    bad = truncation_mismatches(lambda hh: ft.build_features(hh, cfg), hourly, times)
    assert bad == []


def test_truncation_check_catches_a_leaky_feature(cfg, data):
    """Positive control: a centered rolling mean looks ahead and must be caught."""
    hourly, _ = data

    def leaky(hh):
        out = ft.build_features(hh, cfg)
        for sym, f in out.items():
            f["leaky"] = hh[sym]["close"].rolling(5, center=True).mean() / hh[sym]["close"]
        return out

    bad = truncation_mismatches(leaky, hourly, random_times(hourly["BTCUSDT"], 5, seed=2))
    assert bad and all("leaky" in b for b in bad)


def test_no_feature_correlates_suspiciously_with_labels(cfg, data):
    hourly, minutes = data
    feats = ft.build_features(hourly, cfg)
    lc = cfg.labels
    for sym in hourly:
        lab = label_symbol(hourly[sym], minutes[sym], lc.tp_atr, lc.sl_atr, lc.horizon_h, lc.atr_n)
        corr = ft.label_correlations(feats[sym], lab)
        assert corr.notna().all().all()
        flags = ft.leakage_flags(corr, cfg.features.leakage_max_abs_corr)
        assert flags == [], f"{sym}: investigate {flags}\n{corr.loc[flags]}"


def test_correlation_check_catches_future_return(cfg, data):
    """Positive control: a feature built from the realized outcome must be flagged."""
    hourly, minutes = data
    lc = cfg.labels
    h, m = hourly["BTCUSDT"], minutes["BTCUSDT"]
    lab = label_symbol(h, m, lc.tp_atr, lc.sl_atr, lc.horizon_h, lc.atr_n)
    f = ft.compute_features(h, cfg.features)
    f["future_ret"] = h["close"].shift(-12) / h["close"] - 1
    flags = ft.leakage_flags(ft.label_correlations(f, lab), cfg.features.leakage_max_abs_corr)
    assert flags == ["future_ret"]
