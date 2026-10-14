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
