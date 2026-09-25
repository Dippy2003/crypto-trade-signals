"""Download Binance spot 1-minute kline zips from data.binance.vision.

Monthly zips cover the configured range; daily zips cover the current month (and a
recently finished month whose monthly zip is not published yet). Every file is checked
against its ``.CHECKSUM`` (SHA-256). Existing files are skipped.

    python -m src.download [--symbols BTCUSDT ETHUSDT] [--start 2021-01] [--end 2026-08]
"""
from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import Config, ensure_dir, load_config, resolve_path

DOWNLOADED, SKIPPED, MISSING, FAILED = "downloaded", "skipped", "missing", "failed"


@dataclass(frozen=True)
class RemoteFile:
    symbol: str
    name: str        # e.g. BTCUSDT-1m-2024-01.zip
    url: str


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from 'YYYY-MM' to 'YYYY-MM'."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def last_complete_month(today: date) -> str:
    prev = today.replace(day=1) - timedelta(days=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def monthly_file(cfg: Config, symbol: str, y: int, m: int) -> RemoteFile:
    iv = cfg.download.interval
    name = f"{symbol}-{iv}-{y:04d}-{m:02d}.zip"
    return RemoteFile(symbol, name, f"{cfg.download.base_url}/monthly/klines/{symbol}/{iv}/{name}")


def daily_files(cfg: Config, symbol: str, y: int, m: int, until: date) -> list[RemoteFile]:
    """Daily zips for month (y, m) up to and including ``until``."""
    iv = cfg.download.interval
    d, out = date(y, m, 1), []
    while d.month == m and d <= until:
        name = f"{symbol}-{iv}-{d:%Y-%m-%d}.zip"
        out.append(RemoteFile(symbol, name, f"{cfg.download.base_url}/daily/klines/{symbol}/{iv}/{name}"))
        d += timedelta(days=1)
    return out
