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


def test_render_images_fixed_size_and_resumable(project):
    times = assemble_dataset(project, "BTCUSDT").df.index[:6]
    first = ci.render_symbol(project, "BTCUSDT", times, workers=1)
    assert first["rendered"] == 6
    p = ci.image_path(project, "BTCUSDT", times[0])
    assert p.name == f"{times[0]:%Y%m%d_%H}.png"
    img = Image.open(p)
    assert img.size == (project.images.size_px, project.images.size_px)
    colors = {c for _, c in img.convert("RGB").getcolors(100000)}
    assert (255, 255, 255) in colors and len(colors) > 3    # white background plus candle colors
    again = ci.render_symbol(project, "BTCUSDT", times, workers=1)
    assert again["rendered"] == 0 and again["existing"] == 6
