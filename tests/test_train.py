import numpy as np
import pandas as pd
import pytest

from src import train as tr
from src.config import resolve_path
from src.splits import development_mask, make_folds


def test_assemble_dataset(project):
    btc = tr.assemble_dataset(project, "BTCUSDT")
    eth = tr.assemble_dataset(project, "ETHUSDT")
    assert not btc.df[btc.features].isna().any().any()
    assert {"close", "long_label", "short_ret", "entry", "atr"} <= set(btc.df.columns)
    assert "btc_rsi14" in eth.features and "btc_rsi14" not in btc.features
    assert not any(c in btc.features for c in ["close", "entry", "atr", "long_label", "long_ret"])
