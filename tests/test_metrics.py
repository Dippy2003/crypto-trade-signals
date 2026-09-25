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


def test_trade_stats_and_exposure():
    tr = pd.DataFrame({
        "net_ret": [0.02, -0.01, 0.01, -0.01],
        "pnl": [200.0, -100.0, 100.0, -100.0],
        "entry_time": [T, T + pd.Timedelta(hours=1), T + pd.Timedelta(hours=5), T + pd.Timedelta(hours=8)],
        "exit_time": [T + pd.Timedelta(hours=2), T + pd.Timedelta(hours=3), T + pd.Timedelta(hours=6),
                      T + pd.Timedelta(hours=9)],
    })
    s = mt.trade_stats(tr)
    assert s["trades"] == 4 and s["win_rate"] == 0.5
    assert s["avg_win"] == pytest.approx(0.015) and s["avg_loss"] == pytest.approx(-0.01)
    assert s["profit_factor"] == pytest.approx(1.5)
    # covered: 0-3h, 5-6h, 8-9h = 5h of 10h
    assert mt.exposure(tr, T, T + pd.Timedelta(hours=10)) == pytest.approx(0.5)


def test_buy_and_hold(cfg):
    idx = hours(25)
    closes = {"A": pd.Series(np.linspace(100, 110, 25), index=idx), "B": pd.Series(np.linspace(50, 45, 25), index=idx)}
    stats, eq = mt.buy_and_hold(closes, idx[0], idx[-1], cfg)
    expected = (1 - round_trip_cost(cfg)) * (1.10 + 0.90) / 2 - 1
    assert stats["net_return"] == pytest.approx(expected)
    assert stats["exposure"] == 1.0 and eq.index[0] == idx[0]


def test_random_baseline_same_trade_count(cfg):
    from test_backtest import sig
    from src.decision import LONG
    from src.labels import LOSS, WIN
    rows = [sig(h * 13, LONG, WIN if h % 3 == 0 else LOSS) for h in range(40)]
    for r in rows:                       # give both sides outcomes so a random side is meaningful
        r["short_exit"], r["short_minutes"], r["short_label"] = r["long_exit"], 60, r["long_label"]
    pool = pd.DataFrame(rows)
    start, end = pool["time"].min(), pool["time"].max() + pd.Timedelta(hours=13)
    summary, rets = mt.random_baseline(pool, 15, cfg, start, end)
    assert len(rets) == 10 and summary["trades"] <= 15
    assert summary["net_return_p05"] <= summary["net_return"] <= summary["net_return_p95"]
    empty, none = mt.random_baseline(pool, 0, cfg, start, end)
    assert empty["trades"] == 0 and none == []
