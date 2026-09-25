"""Render the last N hourly candles before each decision time as a fixed-style PNG.

Image for bar T shows bars T-(N-1) .. T (all closed at decision time T + 1h), drawn with
mplfinance in one style: candles only, no axes, no indicators, fixed colors, size_px x size_px.
Files are named by bar time (data/images/{SYM}/YYYYMMDD_HH.png) so they join to the same
labels and walk-forward splits as the tabular models. Windows with a missing hour are skipped.
Rendering is parallel and resumable (existing files are kept).

    python -m src.chart_images [--symbols BTCUSDT] [--workers 8] [--limit 1000]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.config import Config, ensure_dir, load_config, resolve_path
from src.data_loader import load_hourly

matplotlib.use("Agg")

_H: pd.DataFrame | None = None       # per-worker hourly data
_IC: dict | None = None               # per-worker image settings


def image_dir(cfg: Config, symbol: str) -> Path:
    return resolve_path(cfg, "images") / symbol


def image_name(t: pd.Timestamp) -> str:
    return f"{t:%Y%m%d_%H}.png"


def image_path(cfg: Config, symbol: str, t: pd.Timestamp) -> Path:
    return image_dir(cfg, symbol) / image_name(t)


def complete_windows(h: pd.DataFrame, times: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Times T whose previous n bars (T-(n-1)h .. T) all exist."""
    pos = h.index.get_indexer(times)
    ok = pos >= n - 1
    first = np.where(ok, pos - (n - 1), 0)
    span = h.index[np.where(ok, pos, 0)] - h.index[first]
    return times[ok & (pos >= 0) & (span == pd.Timedelta(hours=n - 1))]


def _style(ic: dict):
    import mplfinance as mpf

    mc = mpf.make_marketcolors(up=ic["up_color"], down=ic["down_color"], edge="inherit", wick="inherit")
    return mpf.make_mpf_style(marketcolors=mc, facecolor=ic["background"], figcolor=ic["background"],
                              edgecolor=ic["background"], gridstyle="")


def render_window(window: pd.DataFrame, path: Path | None, ic: dict, size_px: int | None = None):
    """Draw candles filling the whole canvas. Returns the figure when ``path`` is None."""
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    size = size_px or ic["size_px"]
    w = window[["open", "high", "low", "close"]].copy()
    w.index = w.index.tz_convert(None)
    lo, hi = w["low"].min(), w["high"].max()
    pad = 0.02 * (hi - lo if hi > lo else hi)
    fig = plt.figure(figsize=(size / ic["dpi"], size / ic["dpi"]), dpi=ic["dpi"])
    ax = fig.add_axes([0, 0, 1, 1])
    mpf.plot(w, type="candle", ax=ax, style=_style(ic), axisoff=True,
             xlim=(-0.5, len(w) - 0.5), ylim=(lo - pad, hi + pad))
    if path is None:
        return fig
    fig.savefig(path, dpi=ic["dpi"], facecolor=ic["background"])
    plt.close(fig)
    return None


def _init(h: pd.DataFrame, ic: dict) -> None:
    global _H, _IC
    _H, _IC = h, ic


def _render_many(jobs: list[tuple[int, str]]) -> int:
    n = _IC["n_candles"]
    for pos, path in jobs:
        render_window(_H.iloc[pos - n + 1: pos + 1], Path(path), _IC)
    return len(jobs)


def render_symbol(cfg: Config, symbol: str, times: pd.DatetimeIndex | None = None,
                  workers: int | None = None, limit: int | None = None) -> dict[str, int]:
    """Render missing images for ``symbol`` (default: every labelled bar). Returns counts."""
    from src.train import assemble_dataset

    ic = dict(cfg.images)
    h = load_hourly(cfg, symbol)[["open", "high", "low", "close"]]
    if times is None:
        times = assemble_dataset(cfg, symbol).df.index
    ok = complete_windows(h, times, ic["n_candles"])
    out = ensure_dir(image_dir(cfg, symbol))
    todo = [t for t in ok if not (out / image_name(t)).exists()]
    existing = len(ok) - len(todo)
    if limit:
        todo = todo[:limit]
    pos = h.index.get_indexer(pd.DatetimeIndex(todo))
    jobs = [(int(p), str(out / image_name(t))) for p, t in zip(pos, todo)]
    workers = workers or ic["workers"]
    chunks = [jobs[i:i + ic["chunk_size"]] for i in range(0, len(jobs), ic["chunk_size"])]
    done = 0
    if workers <= 1 or len(jobs) < ic["chunk_size"]:
        _init(h, ic)
        done = _render_many(jobs)
    else:
        with Pool(workers, initializer=_init, initargs=(h, ic)) as pool:
            for i, n in enumerate(pool.imap_unordered(_render_many, chunks), 1):
                done += n
                if i % 10 == 0 or i == len(chunks):
                    print(f"  {symbol}: {done:,}/{len(jobs):,}")
    counts = {"requested": len(times), "complete_windows": len(ok), "existing": existing, "rendered": done}
    print(f"{symbol}: {counts}")
    return counts
