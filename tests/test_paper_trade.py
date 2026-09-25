import copy

import pandas as pd
import pytest

from src import paper_trade as pt
from src.data_loader import load_minute
from src.labels import LABEL_NAMES, load_labels
from src.live import Signal
from test_live_app import FakeBinance


@pytest.fixture
def cfg(project, tmp_path):
    c = copy.deepcopy(project)
    c.paths.reports = str(tmp_path / "reports")
    c.symbols = ["BTCUSDT"]
    return c


def fake_signal(cfg, lab, t, decision):
    r = lab.loc[t]
    return Signal("BTCUSDT", t, t + pd.Timedelta(hours=1), decision, float(r["entry"]), None, None,
                  0.55 if decision != "NO TRADE" else None, 0.55, 0.2, 0.01, float(r["atr"]), "logreg")


def test_paper_outcome_matches_labels(cfg, monkeypatch):
    lab = load_labels(cfg, "BTCUSDT")
    fake = FakeBinance({}, {"BTCUSDT": load_minute(cfg, "BTCUSDT")})
    times = lab.index[[500, 501, 900]]
    queue = {times[0]: "LONG", times[1]: "SHORT", times[2]: "NO TRADE"}
    current = {}
    monkeypatch.setattr(pt, "get_signal", lambda c, s, k, session=None, now=None:
                        (fake_signal(c, lab, current["t"], queue[current["t"]]), None))

    for t in times[:2]:                      # LONG at bar 500, SHORT one hour later while LONG is open
        current["t"] = t
        pt.run_once(cfg, "logreg", session=fake, now=t + pd.Timedelta(hours=1, seconds=10))
    current["t"] = times[2]
    log = pt.run_once(cfg, "logreg", session=fake, now=times[2] + pd.Timedelta(hours=1, seconds=10))
    log = pt.run_once(cfg, "logreg", session=fake, now=times[2] + pd.Timedelta(hours=1, seconds=10))  # idempotent

    assert list(log["status"]) == ["CLOSED", "SKIPPED", "NO TRADE"]
    row, truth = log.iloc[0], lab.loc[times[0]]
    assert row["entry"] == truth["entry"]
    assert row["exit"] == pytest.approx(truth["long_exit"])
    assert row["minutes"] == truth["long_minutes"]
    assert row["exit_reason"] == {"WIN": "TP", "LOSS": "SL", "NO TRADE": "TIMEOUT"}[LABEL_NAMES[truth["long_label"]]]
    assert row["exit_time"] == times[0] + pd.Timedelta(hours=1, minutes=int(truth["long_minutes"]))
    assert row["pnl"] < row["gross_ret"]                     # costs are charged
    saved = pt.load_log(cfg)
    assert len(saved) == 3 and saved["timestamp"].dt.tz is not None


def test_open_trade_waits_until_resolved(cfg):
    lab = load_labels(cfg, "BTCUSDT")
    t = lab.index[(lab["long_label"] == 1)][5]                # a bar whose long trade times out
    minutes = load_minute(cfg, "BTCUSDT")
    row = pd.Series({"timestamp": t + pd.Timedelta(hours=1), "decision": "LONG", "atr": lab.loc[t, "atr"]})
    early = t + pd.Timedelta(hours=6)
    assert pt.resolve_trade(row, minutes[minutes.index < early], cfg, early) is None
    done = pt.resolve_trade(row, minutes, cfg, t + pd.Timedelta(hours=14))
    assert done["exit_reason"] == "TIMEOUT" and done["minutes"] == 720


def test_summary(cfg):
    log = pd.DataFrame([{"timestamp": pd.Timestamp("2024-05-01", tz="UTC"), "bar_time": pd.Timestamp("2024-04-30 23:00", tz="UTC"),
                         "symbol": "BTCUSDT", "status": "CLOSED", "decision": "LONG", "exit_reason": "TP", "pnl": 0.01}])
    pt.save_log(cfg, log.reindex(columns=pt.LOG_COLS))
    md = pt.summary(cfg, "xgb")
    assert "Paper (live)" in md and "far too few" in md
    assert (pt.resolve_path(cfg, "reports") / "paper_summary.md").exists()
