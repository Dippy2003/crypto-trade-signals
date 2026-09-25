import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src.config import resolve_path
from src.decision import LONG, SHORT


@pytest.fixture(scope="module")
def walk(project):
    return ev.run_walk_forward(project, "logreg")


def test_walk_forward_report(project, walk):
    report = (resolve_path(project, "reports") / "walk_forward_logreg.md").read_text()
    for section in ["## Setup", "## Verdict", "## Results", "## Per fold", "buy & hold", "random entries"]:
        assert section in report
    assert (resolve_path(project, "reports") / "figures" / "equity_walk_forward_logreg.png").exists()
    assert (resolve_path(project, "reports") / "trades_logreg.csv").exists()
    # only calibrated folds are traded
    dec = walk["decisions"]
    assert (dec.loc[~dec["calibrated"], "decision"] == 0).all()


def test_report_with_trades(project, walk):
    dec = walk["decisions"]
    dec = dec[dec["calibrated"]].copy()
    dec.loc[dec.index[::40], "decision"] = LONG
    dec.loc[dec.index[20::40], "decision"] = SHORT
    e = ev.evaluate_decisions(dec, project, "forced")
    e["cfg"] = project
    assert e["model"]["trades"] > 10
    assert e["random"]["trades"] == pytest.approx(e["model"]["trades"], rel=0.5)
    md = ev.walk_forward_markdown(e, ev.per_fold(dec, project), walk["thresholds"], "", "forced", "x.png", 0)
    assert "By symbol and side" in md and "LONG" in md


def test_final_bundle_trained_before_holdout(project, walk):
    b = ev.fit_final(project, "logreg", ["ETHUSDT"])["ETHUSDT"]
    hold = pd.Timestamp(project.splits.holdout_start, tz="UTC")
    assert pd.Timestamp(b["trained_until"]) + pd.Timedelta(hours=1 + project.labels.horizon_h) <= hold
    assert set(b["thresholds"]) == {"long", "short"}
    loaded = ev.load_bundle(project, "logreg", "ETHUSDT")
    from src.features import load_features
    f = load_features(project, "ETHUSDT").dropna().tail(5)
    out = ev.bundle_decide(loaded, f, np.full(len(f), 0.01), project)
    np.testing.assert_allclose(out[["long_c_loss", "long_c_none", "long_c_win"]].sum(axis=1), 1.0)
    assert set(out["decision"]) <= {-1, 0, 1}
