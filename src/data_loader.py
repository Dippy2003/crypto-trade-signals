"""Read Binance 1-minute zips and build clean 1m and 1h candles.

Input : data/raw/{SYM}-1m-YYYY-MM.zip and {SYM}-1m-YYYY-MM-DD.zip (not unzipped)
Output: data/processed/{SYM}_1m.parquet, data/processed/{SYM}_1h.parquet,
        reports/data_quality.md

An hourly bar labelled 10:00 covers 10:00-10:59 and closes at 11:00.

    python -m src.data_loader
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import Config, ensure_dir, load_config, resolve_path

COLS = ["open_time", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close", "volume"]
FILE_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-1m-(?P<date>\d{4}-\d{2}(?:-\d{2})?)\.zip$")


def list_zips(raw: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """Group canonical Binance zip names by symbol. Returns (files_by_symbol, ignored)."""
    by_sym: dict[str, list[Path]] = {}
    ignored = []
    for p in sorted(raw.glob("*.zip")):
        m = FILE_RE.match(p.name)
        if m:
            by_sym.setdefault(m["symbol"], []).append(p)
        else:
            ignored.append(p)
    return by_sym, ignored


def read_zip(path: Path, us_threshold: float) -> pd.DataFrame:
    """Read one zip. Handles files with or without a header and ms or µs timestamps."""
    df = pd.read_csv(path, header=None, usecols=range(6), names=COLS, compression="zip")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()      # drops a header row if any
    t = df["open_time"].astype("int64")
    t = t.where(t < us_threshold, t // 1000)                    # 2025+ files use microseconds
    df["time"] = pd.to_datetime(t, unit="ms", utc=True)
    return df.drop(columns="open_time")


def load_minutes(files: list[Path], us_threshold: float) -> pd.DataFrame:
    """Concatenate, dedupe on time and sort. Index: UTC minute open time."""
    m = (pd.concat([read_zip(f, us_threshold) for f in files])
         .drop_duplicates("time")
         .sort_values("time")
         .set_index("time"))
    return m[PRICE_COLS].astype("float64")


def to_hourly(m: pd.DataFrame, min_minutes: int) -> tuple[pd.DataFrame, int]:
    """Aggregate to 1h bars, dropping hours with fewer than ``min_minutes`` 1m bars."""
    h = m.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    h["n_min"] = m["close"].resample("1h", label="left", closed="left").count()
    keep = h["n_min"] >= min_minutes
    return h[keep], int((~keep).sum())
