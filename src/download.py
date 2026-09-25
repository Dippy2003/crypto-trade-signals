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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_checksum(text: str) -> str:
    """CHECKSUM files look like '<sha256>  <filename>'."""
    return text.strip().split()[0].lower()


class Downloader:
    """Fetches files with retries and checksum verification."""

    def __init__(self, cfg: Config, session: requests.Session | None = None, verify_existing: bool = False):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.verify_existing = verify_existing
        self.raw = ensure_dir(resolve_path(cfg, "raw"))

    def _get(self, url: str) -> bytes | None:
        """GET with retries. Returns None on 404; raises after the last failed attempt."""
        dl = self.cfg.download
        for attempt in range(1, dl.retries + 1):
            try:
                r = self.session.get(url, timeout=dl.timeout_s)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except requests.RequestException:
                if attempt == dl.retries:
                    raise
                time.sleep(dl.backoff_s * attempt)
        return None

    def fetch(self, rf: RemoteFile) -> str:
        """Download one zip and its checksum. Returns a status string."""
        dest = self.raw / rf.name
        chk_path = self.raw / f"{rf.name}.CHECKSUM"
        if dest.exists():
            if not (self.verify_existing or chk_path.exists()):
                return SKIPPED
            try:
                expected = parse_checksum(chk_path.read_text()) if chk_path.exists() else self._checksum(rf)
            except requests.RequestException:
                return FAILED
            if expected is None or sha256_file(dest) == expected:
                return SKIPPED
            print(f"  checksum mismatch on existing {rf.name}, downloading again")

        for attempt in range(1, self.cfg.download.retries + 1):
            try:
                expected = self._checksum(rf)
                if expected is None:
                    return MISSING
                data = self._get(rf.url)
                if data is None:
                    return MISSING
            except requests.RequestException as e:
                print(f"  failed {rf.name}: {e}")
                return FAILED
            if hashlib.sha256(data).hexdigest() == expected:
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                chk_path.write_text(f"{expected}  {rf.name}\n")
                return DOWNLOADED
            print(f"  checksum mismatch on {rf.name} (attempt {attempt})")
            time.sleep(self.cfg.download.backoff_s * attempt)
        return FAILED

    def _checksum(self, rf: RemoteFile) -> str | None:
        data = self._get(rf.url + ".CHECKSUM")
        return None if data is None else parse_checksum(data.decode())
