import hashlib
from datetime import date

import pytest
import requests

from src import download as dl
from src.config import load_config


class FakeResponse:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves bytes from a dict; URLs not in the dict return 404. Can fail the first N calls per URL."""

    def __init__(self, files: dict[str, bytes], flaky: dict[str, int] | None = None, bad: set[str] | None = None):
        self.files, self.flaky, self.bad = files, dict(flaky or {}), bad or set()
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.flaky.get(url, 0) > 0:
            self.flaky[url] -= 1
            raise requests.ConnectionError("temporary failure")
        if url not in self.files:
            return FakeResponse(404)
        body = self.files[url]
        return FakeResponse(200, b"corrupted" if url in self.bad else body)


def serve(cfg, rf: dl.RemoteFile, body: bytes) -> dict[str, bytes]:
    digest = hashlib.sha256(body).hexdigest()
    return {rf.url: body, rf.url + ".CHECKSUM": f"{digest}  {rf.name}\n".encode()}


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={
        "symbols": ["BTCUSDT"],
        "paths": {"raw": str(tmp_path / "raw")},
        "download": {"start": "2024-01", "end": "2024-02", "retries": 3, "backoff_s": 0.0},
    })


def test_month_range_and_last_complete_month():
    assert dl.month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    assert dl.last_complete_month(date(2026, 1, 15)) == "2025-12"


def test_downloads_verifies_and_skips_existing(cfg):
    files = {}
    for m in (1, 2):
        rf = dl.monthly_file(cfg, "BTCUSDT", 2024, m)
        files |= serve(cfg, rf, f"zip-{m}".encode())
    s = FakeSession(files)
    summary = dl.run(cfg, session=s, today=date(2024, 3, 10))
    assert summary[dl.DOWNLOADED] == 2 and summary[dl.FAILED] == 0
    raw = dl.resolve_path(cfg, "raw")
    assert (raw / "BTCUSDT-1m-2024-01.zip").read_bytes() == b"zip-1"
    assert (raw / "BTCUSDT-1m-2024-01.zip.CHECKSUM").exists()

    s2 = FakeSession(files)
    summary = dl.run(cfg, session=s2, today=date(2024, 3, 10))
    assert summary[dl.SKIPPED] == 2
    assert not any(u.endswith(".zip") for u in s2.calls)      # existing zips were not downloaded again


def test_retries_after_connection_error(cfg):
    rf = dl.monthly_file(cfg, "BTCUSDT", 2024, 1)
    s = FakeSession(serve(cfg, rf, b"payload"), flaky={rf.url: 2})
    assert dl.Downloader(cfg, s).fetch(rf) == dl.DOWNLOADED
    assert s.calls.count(rf.url) == 3


def test_checksum_mismatch_is_rejected(cfg):
    rf = dl.monthly_file(cfg, "BTCUSDT", 2024, 1)
    s = FakeSession(serve(cfg, rf, b"payload"), bad={rf.url})
    assert dl.Downloader(cfg, s).fetch(rf) == dl.FAILED
    assert not (dl.resolve_path(cfg, "raw") / rf.name).exists()


def test_missing_file_and_daily_fallback(cfg):
    # 2024-02 monthly is not published yet -> daily files 2024-02-01..03 are used instead
    files = serve(cfg, dl.monthly_file(cfg, "BTCUSDT", 2024, 1), b"jan")
    for rf in dl.daily_files(cfg, "BTCUSDT", 2024, 2, date(2024, 2, 3)):
        files |= serve(cfg, rf, rf.name.encode())
    summary = dl.run(cfg, session=FakeSession(files), today=date(2024, 2, 4))
    assert summary[dl.DOWNLOADED] == 4
    assert (dl.resolve_path(cfg, "raw") / "BTCUSDT-1m-2024-02-03.zip").exists()


def test_daily_files_for_current_month(cfg):
    cfg.download.end = None
    files = {}
    for m in (1, 2):
        files |= serve(cfg, dl.monthly_file(cfg, "BTCUSDT", 2024, m), b"m")
    for rf in dl.daily_files(cfg, "BTCUSDT", 2024, 3, date(2024, 3, 4)):
        files |= serve(cfg, rf, b"d")
    summary = dl.run(cfg, session=FakeSession(files), today=date(2024, 3, 5))
    assert summary[dl.DOWNLOADED] == 2 + 4
