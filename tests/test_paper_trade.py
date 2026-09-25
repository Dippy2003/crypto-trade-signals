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


def test_open_trade_waits_until_resolved(cfg):
    lab = load_labels(cfg, "BTCUSDT")
    t = lab.index[(lab["long_label"] == 1)][5]                # a bar whose long trade times out
    minutes = load_minute(cfg, "BTCUSDT")
    row = pd.Series({"timestamp": t + pd.Timedelta(hours=1), "decision": "LONG", "atr": lab.loc[t, "atr"]})
    early = t + pd.Timedelta(hours=6)
    assert pt.resolve_trade(row, minutes[minutes.index < early], cfg, early) is None
    done = pt.resolve_trade(row, minutes, cfg, t + pd.Timedelta(hours=14))
    assert done["exit_reason"] == "TIMEOUT" and done["minutes"] == 720
