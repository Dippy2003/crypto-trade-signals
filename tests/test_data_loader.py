import pandas as pd
import pytest

from src import data_loader as dlr
from src.config import load_config, resolve_path
from synth import random_walk_minutes, to_binance, write_months, write_zip

US = 1.0e14


def test_ms_and_us_files_mixed(tmp_path):
    m = random_walk_minutes("2024-12-31 22:00", 240, seed=1)   # crosses into 2025
    files = write_months(tmp_path, "BTCUSDT", m)
    assert [f.name for f in files] == ["BTCUSDT-1m-2024-12.zip", "BTCUSDT-1m-2025-01.zip"]
    assert pd.read_csv(files[1], header=None)[0].iloc[0] > US       # really microseconds
    got = dlr.load_minutes(files, US)
    pd.testing.assert_index_equal(got.index, m.index)
    assert str(got.index.tz) == "UTC"
    pd.testing.assert_frame_equal(got, m, check_freq=False)


def test_duplicates_removed_and_sorted(tmp_path):
    m = random_walk_minutes("2024-03-01", 180, seed=2)
    rows = to_binance(m)
    shuffled_with_dupes = pd.concat([rows, rows.iloc[10:40]]).sample(frac=1, random_state=0)
    a = write_zip(tmp_path / "BTCUSDT-1m-2024-03.zip", shuffled_with_dupes)
    b = write_zip(tmp_path / "BTCUSDT-1m-2024-03-01.zip", rows.iloc[:60])   # daily file overlapping monthly
    got = dlr.load_minutes([a, b], US)
    assert got.index.is_unique and got.index.is_monotonic_increasing
    assert len(got) == 180


def test_header_row_is_dropped(tmp_path):
    m = random_walk_minutes("2024-03-01", 60, seed=3)
    f = write_zip(tmp_path / "BTCUSDT-1m-2024-03.zip", to_binance(m), header=True)
    assert len(dlr.load_minutes([f], US)) == 60


def test_missing_minutes_and_largest_gap(tmp_path):
    m = random_walk_minutes("2024-03-01", 600, seed=4)
    gap = m.index[100:130]                  # 30-minute hole
    small = m.index[[300, 450]]             # two single missing minutes
    m = m.drop(gap.union(small))
    h, dropped = dlr.to_hourly(m, 50)
    s = dlr.quality_stats(m, h, dropped)
    assert s["missing_1m"] == 32
    assert s["largest_gap_min"] == 30
    assert s["largest_gap_start"] == gap[0]
