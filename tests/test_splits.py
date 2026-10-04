import numpy as np
import pandas as pd
import pytest

from src import splits as sp
from src.config import load_config


@pytest.fixture
def cfg():
    return load_config(overrides={"splits": {"first_test_start": "2024-01-01", "test_months": 3,
                                             "holdout_start": "2025-01-01", "embargo_h": 24,
                                             "val_fraction": 0.15}})


def test_folds_are_consecutive_and_stop_at_holdout(cfg):
    folds = sp.make_folds(cfg)
    assert [f.k for f in folds] == [1, 2, 3, 4]
    assert folds[0].test_start == pd.Timestamp("2024-01-01", tz="UTC")
    assert folds[-1].test_end == pd.Timestamp("2025-01-01", tz="UTC")
    for a, b in zip(folds[:-1], folds[1:]):
        assert a.test_end == b.test_start
