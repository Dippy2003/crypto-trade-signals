import numpy as np
import pandas as pd
import pytest

from src import labels as lb
from src.data_loader import to_hourly
from synth import random_walk_minutes


def reference_labels(h, m, tp_atr, sl_atr, horizon_h, atr_n):
    """The original make_labels.py loop, kept as ground truth for the vectorized version."""
    H = horizon_h * 60
    atr = lb.wilder_atr(h, atr_n).to_numpy()
    mt = m.index.values
    mo, mh, ml, mc = (m[c].to_numpy() for c in ["open", "high", "low", "close"])
    decision = (h.index + pd.Timedelta(hours=1)).values
    start = np.searchsorted(mt, decision)
    rows = []
    for i in range(len(h)):
        s, a = start[i], atr[i]
        e = s + H
        if np.isnan(a) or e > len(mt) or mt[s] != decision[i] or mt[e - 1] - mt[s] != (H - 1) * lb.ONE_MIN:
            continue
        entry = mo[s]
        hi, lo = mh[s:e], ml[s:e]
        L = lb.barrier_outcome(hi, lo, mc[e - 1], entry, entry + tp_atr * a, entry - sl_atr * a, +1)
        S = lb.barrier_outcome(hi, lo, mc[e - 1], entry, entry - tp_atr * a, entry + sl_atr * a, -1)
        rows.append((h.index[i], entry, a, *L, *S))
    return pd.DataFrame(rows, columns=[
        "time", "entry", "atr", "long_label", "long_ret", "long_minutes",
        "short_label", "short_ret", "short_minutes"]).set_index("time")


def test_vectorized_matches_reference_loop():
    m = random_walk_minutes("2024-01-01", 60 * 24 * 6, seed=11, vol=0.002)
    m = m.drop(m.index[3000:3007]).drop(m.index[5000:5100])   # gaps: skipped windows and a dropped hour
    h, _ = to_hourly(m, 50)
    ref = reference_labels(h, m, 2.0, 1.0, 12, 14)
    new = lb.label_symbol(h, m, 2.0, 1.0, 12, 14, chunk_rows=7)   # odd chunk size exercises the chunking
    assert len(ref) > 50 and len(ref) < len(h)
    pd.testing.assert_index_equal(new.index, ref.index)
    for c in ref.columns:
        np.testing.assert_array_equal(new[c].to_numpy(), ref[c].to_numpy(), err_msg=c)
    assert set(new["long_label"]) == {lb.LOSS, lb.NO_TRADE, lb.WIN}


def test_exit_price_consistent_with_return():
    m = random_walk_minutes("2024-01-01", 60 * 24 * 3, seed=12, vol=0.002)
    h, _ = to_hourly(m, 50)
    lab = lb.label_symbol(h, m, 2.0, 1.0, 12, 14)
    np.testing.assert_allclose(lab["long_exit"] / lab["entry"] - 1, lab["long_ret"])
    np.testing.assert_allclose(1 - lab["short_exit"] / lab["entry"], lab["short_ret"])


def test_label_distribution_sums_to_100():
    m = random_walk_minutes("2024-12-28", 60 * 24 * 8, seed=13, vol=0.002)
    h, _ = to_hourly(m, 50)
    dist = lb.label_distribution({"BTCUSDT": lb.label_symbol(h, m, 2.0, 1.0, 12, 14)})
    assert set(dist["year"]) == {2024, 2025}
    np.testing.assert_allclose(dist[["LOSS", "NO TRADE", "WIN"]].sum(axis=1), 100)


# ---------------------------------------------------------------------------
# Hand-built paths. Hourly bars have range 1.0 so ATR(14) = 1.0; price sits at 100.
# LONG: TP 102, SL 99.   SHORT: TP 98, SL 101.   Horizon 12h = 720 minutes.
# ---------------------------------------------------------------------------
H = 720
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def hand_case(edit=None, entry_open=100.0, last_close=100.0, drop=()):
    """Return (labels, bar_time). Only bar 15 has a complete minute window."""
    idx = pd.date_range(T0, periods=20, freq="1h", tz="UTC")
    h = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx)
    bar = idx[15]
    mi = pd.date_range(bar + pd.Timedelta(hours=1), periods=H, freq="1min", tz="UTC")
    m = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=mi)
    m.iloc[0, m.columns.get_loc("open")] = entry_open
    m.iloc[-1, m.columns.get_loc("close")] = last_close
    for minute, col, value in (edit or []):
        m.iloc[minute, m.columns.get_loc(col)] = value
    m = m.drop(m.index[list(drop)])
    return lb.label_symbol(h, m, 2.0, 1.0, 12, 14), bar


def test_atr_is_one_for_hand_built_bars():
    lab, bar = hand_case()
    assert list(lab.index) == [bar]
    assert lab.loc[bar, "atr"] == pytest.approx(1.0)


def test_take_profit_hit_first():
    lab, bar = hand_case([(10, "high", 102.5), (50, "low", 98.5)])
    r = lab.loc[bar]
    assert r["long_label"] == lb.WIN and r["long_minutes"] == 11
    assert r["long_ret"] == pytest.approx(0.02) and r["long_exit"] == pytest.approx(102)


def test_stop_loss_hit_first():
    lab, bar = hand_case([(5, "low", 98.9), (20, "high", 103.0)])
    r = lab.loc[bar]
    assert r["long_label"] == lb.LOSS and r["long_minutes"] == 6
    assert r["long_ret"] == pytest.approx(-0.01) and r["long_exit"] == pytest.approx(99)


def test_both_barriers_in_same_minute_is_loss():
    lab, bar = hand_case([(7, "high", 102.5), (7, "low", 97.5)])
    r = lab.loc[bar]
    assert r["long_label"] == lb.LOSS and r["long_minutes"] == 8
    assert r["short_label"] == lb.LOSS and r["short_minutes"] == 8
    assert r["long_ret"] == pytest.approx(-0.01) and r["short_ret"] == pytest.approx(-0.01)


def test_timeout_is_no_trade():
    lab, bar = hand_case([(100, "high", 100.9), (200, "low", 99.1)], last_close=100.3)
    r = lab.loc[bar]
    assert r["long_label"] == lb.NO_TRADE and r["short_label"] == lb.NO_TRADE
    assert r["long_minutes"] == H and r["short_minutes"] == H
    assert r["long_ret"] == pytest.approx(0.003) and r["short_ret"] == pytest.approx(-0.003)
    assert r["long_exit"] == pytest.approx(100.3)


def test_gap_in_window_skips_row():
    lab, _ = hand_case(drop=[300])
    assert lab.empty


def test_missing_entry_minute_skips_row():
    lab, _ = hand_case(drop=[0])
    assert lab.empty


def test_short_side_is_mirrored():
    lab, bar = hand_case([(30, "low", 97.9)])
    r = lab.loc[bar]
    assert r["short_label"] == lb.WIN and r["short_minutes"] == 31
    assert r["short_ret"] == pytest.approx(0.02) and r["short_exit"] == pytest.approx(98)
    assert r["long_label"] == lb.LOSS and r["long_minutes"] == 31      # 97.9 is also below the long stop

    lab, bar = hand_case([(40, "high", 101.2)])
    r = lab.loc[bar]
    assert r["short_label"] == lb.LOSS and r["short_ret"] == pytest.approx(-0.01)
    assert r["long_label"] == lb.NO_TRADE


def test_entry_is_first_minute_after_bar_close():
    lab, bar = hand_case(entry_open=100.2)
    r = lab.loc[bar]
    assert r["entry"] == 100.2                               # open of bar close + 0 min, not the bar close
    lab2, _ = hand_case([(0, "high", 102.3)], entry_open=100.2)
    assert lab2.loc[bar, "long_label"] == lb.WIN             # entry minute's own high counts ...
    assert lab2.loc[bar, "long_minutes"] == 1                # ... and is minute 1 of the trade
    lab3, _ = hand_case([(0, "high", 102.1)], entry_open=100.2)
    assert lab3.loc[bar, "long_label"] == lb.NO_TRADE        # TP is 102.2 because entry is 100.2


def test_barrier_outcome_single_trade():
    hi = np.array([100.0, 101.0, 102.5])
    lo = np.array([99.5, 99.5, 99.5])
    assert lb.barrier_outcome(hi, lo, 102.0, 100.0, 102.0, 99.0, +1) == (lb.WIN, pytest.approx(0.02), 3)
    assert lb.barrier_outcome(hi, lo, 102.0, 100.0, 98.0, 101.0, -1) == (lb.LOSS, pytest.approx(-0.01), 2)
