import numpy as np
import pandas as pd
import pytest

from src import backtest as bt
from src.config import load_config
from src.decision import LONG, SHORT, net_return
from src.labels import LOSS, NO_TRADE, WIN

T = pd.Timestamp("2024-05-01 00:00", tz="UTC")


@pytest.fixture
def cfg():
    return load_config()


def sig(hour, decision, label, symbol="BTCUSDT", minutes=60, entry=100.0, atr=1.0, day=0):
    """One signal row; exit price follows the label (TP 2 ATR, SL 1 ATR, timeout +0.1)."""
    side = "long" if decision == LONG else "short"
    s = 1 if decision == LONG else -1
    exit_px = {WIN: entry + s * 2 * atr, LOSS: entry - s * atr, NO_TRADE: entry + 0.1}[label]
    row = {"symbol": symbol, "time": T + pd.Timedelta(days=day, hours=hour), "decision": decision,
           "entry": entry, "atr": atr, "confidence": 0.5}
    for sd in ("long", "short"):
        row[f"{sd}_exit"], row[f"{sd}_minutes"], row[f"{sd}_label"] = entry, 720, NO_TRADE
    row[f"{side}_exit"], row[f"{side}_minutes"], row[f"{side}_label"] = exit_px, minutes, label
    return row


def run(rows, cfg):
    return bt.backtest(pd.DataFrame(rows), cfg)


def test_long_win_sizing_and_log(cfg):
    res = run([sig(0, LONG, WIN, minutes=90)], cfg)
    t = res.trades.iloc[0]
    # risk 1% of 10,000 over a 1% stop -> 10,000 notional
    assert t["notional"] == pytest.approx(10_000)
    assert t["entry_time"] == T + pd.Timedelta(hours=1)
    assert t["exit_time"] == T + pd.Timedelta(hours=1, minutes=90)
    assert (t["tp"], t["sl"], t["exit"], t["exit_reason"]) == (102.0, 99.0, 102.0, "TP")
    assert t["pnl"] == pytest.approx(10_000 * net_return(1, 100.0, 102.0, cfg))
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t["pnl"])


def test_stop_loss_loses_about_one_percent(cfg):
    res = run([sig(0, SHORT, LOSS)], cfg)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "SL" and t["sl"] == 101.0 and t["side"] == "SHORT"
    assert t["pnl"] / 10_000 == pytest.approx(-0.01 - 0.0024, abs=3e-4)


def test_leverage_cap(cfg):
    res = run([sig(0, LONG, WIN, atr=0.1)], cfg)       # 0.1% stop would need 10x notional
    assert res.trades.iloc[0]["notional"] == pytest.approx(2 * 10_000)
