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


def test_development_data_excludes_holdout(project):
    ds = tr.development_data(tr.assemble_dataset(project, "BTCUSDT"), project)
    assert development_mask(ds.df.index, project).all()
    assert ds.df.index.max() < pd.Timestamp(project.splits.holdout_start, tz="UTC")


def test_fold_metrics_known_values():
    y = np.array([0, 2, 1, 2])
    p = np.array([[.8, .1, .1], [.1, .1, .8], [.2, .6, .2], [.6, .2, .2]])
    m = tr.fold_metrics(y, p, prior=np.array([1 / 3] * 3))
    assert m["prior_log_loss"] == pytest.approx(np.log(3))
    assert m["recall_WIN"] == 0.5 and m["precision_WIN"] == 1.0
    assert m["confusion"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]


def test_logreg_walk_forward(project):
    oof = tr.run(project, "logreg")
    folds = {f.k: f for f in make_folds(project)}
    assert sorted(oof["fold"].unique()) == [1, 2, 3]
    for k, g in oof.groupby("fold"):
        assert g["time"].min() >= folds[k].test_start and g["time"].max() < folds[k].test_end
    for side in tr.SIDES:
        p = oof[[f"{side}_{c}" for c in tr.PROBA]].to_numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert set(oof["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert (resolve_path(project, "reports") / "train_logreg.md").exists()
    pd.testing.assert_frame_equal(tr.load_oof(project, "logreg"), oof)
    # labels in the OOF file are the real labels for those rows
    lab = tr.assemble_dataset(project, "BTCUSDT").df
    b = oof[oof["symbol"] == "BTCUSDT"].set_index("time")
    assert (lab.loc[b.index, "long_label"].to_numpy() == b["long_label"].to_numpy()).all()


def test_class_weights_balance_classes():
    y = np.array([0] * 6 + [1] * 3 + [2] * 1)
    w = tr.class_weights(y)
    assert [w[y == c].sum() for c in (0, 1, 2)] == pytest.approx([10 / 3] * 3)
