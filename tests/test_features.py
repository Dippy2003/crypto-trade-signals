import numpy as np
import pandas as pd
import pytest

from src import features as ft
from src.config import load_config
from src.data_loader import to_hourly
from synth import random_walk_minutes


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def hourly():
    btc, _ = to_hourly(random_walk_minutes("2024-01-01", 60 * 24 * 20, seed=21, price=40000), 50)
    eth, _ = to_hourly(random_walk_minutes("2024-01-01", 60 * 24 * 20, seed=22, price=2500), 50)
    return {"BTCUSDT": btc, "ETHUSDT": eth}


def test_expected_columns(cfg, hourly):
    f = ft.compute_features(hourly["BTCUSDT"], cfg.features)
    for col in ["ret_1h", "ret_24h", "close_ema20", "ema20_ema50", "ema50_ema200", "rsi14", "macd",
                "macd_signal", "macd_hist", "bb_pctb", "bb_width", "atr14_close", "rv_6h", "rv_24h",
                "hl_range", "co_range", "vol_rel20", "hour_sin", "dow_cos"]:
        assert col in f.columns, col
    assert f.index.equals(hourly["BTCUSDT"].index)
    complete = f.dropna()
    assert len(complete) > 200                       # EMA200 warm-up leaves most rows complete
    assert complete["rsi14"].between(0, 1).all()


def test_scale_free(cfg, hourly):
    h = hourly["BTCUSDT"]
    scaled = h.copy()
    scaled[["open", "high", "low", "close"]] *= 1000.0
    a = ft.compute_features(h, cfg.features)
    b = ft.compute_features(scaled, cfg.features)
    pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-12)


def test_time_features(cfg, hourly):
    f = ft.compute_features(hourly["BTCUSDT"], cfg.features)
    t = pd.Timestamp("2024-01-03 06:00", tz="UTC")        # a Wednesday, 06:00
    assert f.loc[t, "hour_sin"] == pytest.approx(1.0)
    assert f.loc[t, "dow_sin"] == pytest.approx(np.sin(2 * np.pi * 2 / 7))


def test_context_columns_only_for_other_symbols(cfg, hourly):
    feats = ft.build_features(hourly, cfg)
    btc, eth = feats["BTCUSDT"], feats["ETHUSDT"]
    assert not any(c.startswith("btc_") for c in btc.columns)
    assert "btc_ret_1h" in eth.columns and "btc_hour_sin" not in eth.columns
    pd.testing.assert_series_equal(eth["btc_rsi14"], btc["rsi14"].reindex(eth.index), check_names=False)


def test_dropped_hour_does_not_shift_returns(cfg, hourly):
    h = hourly["BTCUSDT"].drop(hourly["BTCUSDT"].index[300])
    f = ft.compute_features(h, cfg.features)
    after = h.index[300]                                   # the bar right after the missing hour
    assert np.isnan(f.loc[after, "ret_1h"])                # its 1h-ago bar does not exist
    assert f.loc[after, "ret_2h"] == pytest.approx(h.loc[after, "close"] / h["close"].iloc[299] - 1)
