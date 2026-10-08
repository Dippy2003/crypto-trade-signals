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


def test_one_position_per_symbol(cfg):
    rows = [sig(0, LONG, WIN, minutes=180),            # open 01:00 -> 04:00
            sig(1, SHORT, LOSS),                       # 02:00 same symbol: skipped
            sig(1, LONG, WIN, symbol="ETHUSDT"),       # other symbol: allowed
            sig(3, LONG, NO_TRADE)]                    # 04:00 entry: first position just closed
    res = run(rows, cfg)
    assert list(res.trades["symbol"]) == ["BTCUSDT", "ETHUSDT", "BTCUSDT"]
    assert res.skipped["position_open"] == 1
    assert res.trades.iloc[2]["exit_reason"] == "TIMEOUT"


def test_sizing_uses_realized_equity(cfg):
    res = run([sig(0, LONG, WIN, minutes=30), sig(2, LONG, WIN)], cfg)
    first, second = res.trades.iloc[0], res.trades.iloc[1]
    assert second["equity_at_entry"] == pytest.approx(10_000 + first["pnl"])
    assert second["notional"] == pytest.approx(second["equity_at_entry"])


def test_max_daily_loss_stop(cfg):
    rows = [sig(h, LONG, LOSS, minutes=10) for h in range(5)]        # five stop-outs in one day
    rows.append(sig(0, LONG, WIN, day=1))                            # next day trades again
    res = run(rows, cfg)
    # each loss is ~1.24%; after 3 losses (> 3% of the day's start) trading stops for the day
    assert (res.trades["signal_time"].dt.day == 1).sum() == 3
    assert res.skipped["daily_loss_stop"] == 2
    assert res.trades.iloc[-1]["signal_time"] == T + pd.Timedelta(days=1)


def test_no_trades_gives_flat_equity(cfg):
    rows = [sig(0, LONG, WIN)]
    rows[0]["decision"] = 0
    res = run(rows, cfg)
    assert res.trades.empty and (res.equity == 10_000).all()


def test_save_trades(cfg, tmp_path):
    res = run([sig(0, LONG, WIN), sig(5, SHORT, LOSS)], cfg)
    bt.save_trades(res, tmp_path / "t.csv")
    back = pd.read_csv(tmp_path / "t.csv")
    assert list(back["exit_reason"]) == ["TP", "SL"]
    assert {"symbol", "side", "entry", "tp", "sl", "exit", "pnl", "net_ret", "confidence"} <= set(back.columns)
    assert np.isclose(back["pnl"].sum(), res.trades["pnl"].sum())
