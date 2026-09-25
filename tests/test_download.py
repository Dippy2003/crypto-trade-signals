import hashlib
from datetime import date

import pytest
import requests

from src import download as dl
from src.config import load_config


def test_month_range_and_last_complete_month():
    assert dl.month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    assert dl.last_complete_month(date(2026, 1, 15)) == "2025-12"
