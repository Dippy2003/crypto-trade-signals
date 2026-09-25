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
