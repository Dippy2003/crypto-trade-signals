"""
Step 1: Read Binance 1-minute zip files -> clean 1m data + 1h candles.

Input : data/raw/BTCUSDT-1m-2024-01.zip ... (don't unzip)
Output: data/processed/BTCUSDT_1m.parquet
        data/processed/BTCUSDT_1h.parquet
"""
import glob
import os

import pandas as pd

RAW = "data/raw"
OUT = "data/processed"
MIN_MINUTES_PER_HOUR = 50   # drop hours with too many missing minutes

COLS = ["open_time", "open", "high", "low", "close", "volume"]
os.makedirs(OUT, exist_ok=True)


def read_zip(path):
    df = pd.read_csv(path, header=None, usecols=range(6), names=COLS, compression="zip")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()      # drops a header row if any
    t = df["open_time"].astype("int64")
    t = t.where(t < 1e14, t // 1000)                            # 2025+ files use microseconds
    df["time"] = pd.to_datetime(t, unit="ms", utc=True)
    return df.drop(columns="open_time")


files = sorted(glob.glob(f"{RAW}/*-1m-*.zip"))
if not files:
    raise SystemExit(f"No zip files found in {RAW}/")

symbols = sorted({os.path.basename(f).split("-")[0] for f in files})

for sym in symbols:
    sym_files = [f for f in files if os.path.basename(f).startswith(sym + "-")]
    print(f"\n{sym}: reading {len(sym_files)} files...")

    m = (pd.concat([read_zip(f) for f in sym_files])
           .drop_duplicates("time")
           .sort_values("time")
           .set_index("time"))

    # Data-quality report
    expected = int((m.index[-1] - m.index[0]).total_seconds() // 60) + 1
    print(f"  1m bars : {len(m):,}  ({m.index[0]} -> {m.index[-1]})")
    print(f"  missing : {expected - len(m):,} minutes ({100 * (1 - len(m) / expected):.2f}%)")

    # 1-hour candles. Bar labelled 10:00 covers 10:00-10:59 and closes at 11:00.
    h = m.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    h["n_min"] = m["close"].resample("1h", label="left", closed="left").count()
    dropped = (h["n_min"] < MIN_MINUTES_PER_HOUR).sum()
    h = h[h["n_min"] >= MIN_MINUTES_PER_HOUR]
    print(f"  1h bars : {len(h):,}  (dropped {dropped} incomplete hours)")

    m.to_parquet(f"{OUT}/{sym}_1m.parquet")
    h.to_parquet(f"{OUT}/{sym}_1h.parquet")
    print(f"  saved   : {OUT}/{sym}_1m.parquet, {OUT}/{sym}_1h.parquet")
