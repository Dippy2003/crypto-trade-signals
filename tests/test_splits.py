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


@pytest.fixture
def times():
    t = pd.date_range("2023-01-01", "2025-06-30 23:00", freq="1h", tz="UTC", name="time")
    return t.delete(np.arange(5000, 5010))               # a small hole, as in real data


def test_folds_are_consecutive_and_stop_at_holdout(cfg):
    folds = sp.make_folds(cfg)
    assert [f.k for f in folds] == [1, 2, 3, 4]
    assert folds[0].test_start == pd.Timestamp("2024-01-01", tz="UTC")
    assert folds[-1].test_end == pd.Timestamp("2025-01-01", tz="UTC")
    for a, b in zip(folds[:-1], folds[1:]):
        assert a.test_end == b.test_start


def test_train_is_purged_and_embargoed(cfg, times):
    emb = pd.Timedelta(hours=24)
    for f in sp.make_folds(cfg):
        tr, te = sp.fold_masks(times, f, cfg)
        assert not (tr & te).any()
        train_t, test_t = times[tr], times[te]
        assert train_t.max() < test_t.min()
        # every training label window closes at least one embargo before the test starts
        assert sp.label_end(train_t, cfg).max() <= f.test_start - emb
        # and the purge is tight: the next hourly bar would violate it
        assert sp.label_end(train_t, cfg).max() + pd.Timedelta(hours=1) > f.test_start - emb
        assert test_t.min() >= f.test_start and test_t.max() < f.test_end


def test_expanding_window(cfg, times):
    sizes = [sp.fold_masks(times, f, cfg)[0].sum() for f in sp.make_folds(cfg)]
    assert sizes == sorted(sizes) and sizes[0] > 0


def test_nothing_in_development_touches_the_holdout(cfg, times):
    hold = pd.Timestamp("2025-01-01", tz="UTC")
    dev = sp.development_mask(times, cfg)
    assert (sp.label_end(times[dev], cfg) <= hold - pd.Timedelta(hours=24)).all()
    for f in sp.make_folds(cfg):
        tr, te = sp.fold_masks(times, f, cfg)
        assert not (tr & ~dev).any() and not (te & ~dev).any()
    ho = sp.holdout_mask(times, cfg)
    assert times[ho].min() == hold and not (ho & dev).any()
    with pytest.raises(RuntimeError):
        sp.assert_no_holdout(times, cfg)
    sp.assert_no_holdout(times[dev], cfg)


def test_validation_tail_is_purged(cfg, times):
    tr, _ = sp.fold_masks(times, sp.make_folds(cfg)[2], cfg)
    fit, val = sp.fit_val_masks(times, tr, cfg)
    assert not (fit & val).any()
    assert val.sum() == round(tr.sum() * 0.15)
    val_start = times[val].min()
    assert sp.label_end(times[fit], cfg).max() <= val_start - pd.Timedelta(hours=24)
    assert times[fit].max() < val_start
