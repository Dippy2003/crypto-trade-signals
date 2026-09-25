import pandas as pd
from PIL import Image

from src import chart_images as ci
from src.train import assemble_dataset


def test_complete_windows_skip_gaps():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC").delete(70)
    h = pd.DataFrame({"close": 1.0}, index=idx)
    ok = ci.complete_windows(h, idx, 60)
    # windows ending at hours 59..69 are complete; every later window contains the missing hour 70
    assert list(ok) == list(idx[59:70])
