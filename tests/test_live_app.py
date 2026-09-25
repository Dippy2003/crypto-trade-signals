import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src import live
from src import train as tr
from src.data_loader import load_hourly, load_minute
from src.features import load_features


class FakeBinance:
    """Serves /api/v3/klines from local frames, honouring interval, startTime, endTime and limit."""

    def __init__(self, hourly: dict, minutes: dict | None = None):
        self.data = {"1h": hourly, "1m": minutes or {}}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        df = self.data[params["interval"]][params["symbol"]]
        step = 3_600_000 if params["interval"] == "1h" else 60_000
        ot = (df.index.astype("int64") // 1_000_000).to_numpy()
        sel = np.ones(len(df), dtype=bool)
        if "startTime" in params:
            sel &= ot >= params["startTime"]
        if "endTime" in params:
            sel &= ot <= params["endTime"]
        idx = np.flatnonzero(sel)
        idx = idx[:params["limit"]] if "startTime" in params else idx[-params["limit"]:]
        rows = [[int(ot[i]), *(str(v) for v in df.iloc[i][["open", "high", "low", "close", "volume"]]),
                 int(ot[i]) + step - 1, "0", 1, "0", "0", "0"] for i in idx]
        return _Resp(rows)


class _Resp:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return self.rows


@pytest.fixture(scope="module")
def bundle(project):
    tr.run(project, "logreg")
    return ev.fit_final(project, "logreg")


def test_fetch_hourly_paginates_and_drops_open_candle(project):
    cfg = project
    h = load_hourly(cfg, "BTCUSDT")
    fake = FakeBinance({"BTCUSDT": h})
    now = h.index[-1] + pd.Timedelta(minutes=30)             # last bar is still forming
    old = cfg.live.request_limit
    cfg.live.request_limit = 400
    try:
        got = live.fetch_hourly("BTCUSDT", cfg, bars=1000, session=fake, now=now)
    finally:
        cfg.live.request_limit = old
    assert fake.calls >= 3
    assert len(got) == 1000 and got.index[-1] == h.index[-2]
    pd.testing.assert_frame_equal(got, h.iloc[-1001:-1][["open", "high", "low", "close", "volume"]], check_freq=False)


def test_fetch_minutes_forward(project):
    m = load_minute(project, "BTCUSDT")
    fake = FakeBinance({}, {"BTCUSDT": m})
    start = m.index[1000]
    got = live.fetch_minutes("BTCUSDT", project, start, start + pd.Timedelta(minutes=2500), session=fake)
    assert len(got) == 2500 and got.index[0] == start


def test_live_features_match_training(project, bundle):
    """Signal built from REST-style candles uses exactly the training features."""
    cfg = project
    hourly = {s: load_hourly(cfg, s) for s in ["BTCUSDT", "ETHUSDT"]}
    t = pd.Timestamp("2024-06-10 12:00", tz="UTC")
    cut = {s: h[h.index <= t][["open", "high", "low", "close", "volume"]] for s, h in hourly.items()}
    sig = live.make_signal(cfg, "ETHUSDT", cut, bundle["ETHUSDT"])
    assert sig.bar_time == t and sig.decision_time == t + pd.Timedelta(hours=1)
    train_row = load_features(cfg, "ETHUSDT").loc[[t]]
    direct = ev.bundle_decide(bundle["ETHUSDT"], train_row, np.array([sig.atr / sig.entry]), cfg).iloc[0]
    assert sig.p_long_win == pytest.approx(direct["long_c_win"])
    assert sig.p_short_win == pytest.approx(direct["short_c_win"])
    assert sig.decision in ("LONG", "SHORT", "NO TRADE")
    text = live.format_signal(sig)
    assert "ETHUSDT" in text and sig.decision in text


def test_signal_levels_for_long(project, bundle):
    cfg = project
    hourly = {s: load_hourly(cfg, s)[["open", "high", "low", "close", "volume"]] for s in ["BTCUSDT"]}
    b = dict(bundle["BTCUSDT"], thresholds={"long": 0.0, "short": np.inf})
    cfg.decision.timeout_return = 1.0                        # makes EV positive so LONG is chosen
    try:
        sig = live.make_signal(cfg, "BTCUSDT", hourly, b)
    finally:
        cfg.decision.timeout_return = 0.0
    assert sig.decision == "LONG"
    assert sig.tp == pytest.approx(sig.entry + 2 * sig.atr) and sig.sl == pytest.approx(sig.entry - sig.atr)
    assert "TP" in live.format_signal(sig)


def test_get_signal_end_to_end(project, bundle):
    hourly = {s: load_hourly(project, s) for s in ["BTCUSDT", "ETHUSDT"]}
    now = hourly["BTCUSDT"].index[-1] + pd.Timedelta(hours=1, seconds=5)
    sig, h = live.get_signal(project, "ETHUSDT", "logreg", session=FakeBinance(hourly), now=now)
    assert sig.bar_time == hourly["ETHUSDT"].index[-1]
    assert len(h) == min(project.live.history_bars, len(hourly["ETHUSDT"]))


def test_app_renders_without_network():
    from streamlit.testing.v1 import AppTest

    from src.config import ROOT

    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Crypto trade signals"
    assert [s.label for s in at.selectbox] == ["Symbol", "Model"]
