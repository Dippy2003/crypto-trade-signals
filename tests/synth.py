"""Small synthetic market data in Binance format for tests (never the real data)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BINANCE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def random_walk_minutes(start: str, n: int, seed: int = 0, price: float = 100.0,
                        vol: float = 0.001) -> pd.DataFrame:
    """1-minute OHLCV random walk indexed by UTC open time."""
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, (2, n)))
    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * (1 + wick[0]),
        "low": np.minimum(open_, close) * (1 - wick[1]),
        "close": close,
        "volume": rng.uniform(1, 10, n),
    }, index=pd.date_range(start, periods=n, freq="1min", tz="UTC", name="time"))
    return df


def to_binance(df: pd.DataFrame, unit: str = "ms") -> pd.DataFrame:
    """Binance kline rows (12 columns). ``unit`` is 'ms' (before 2025) or 'us' (2025+)."""
    scale = 1_000 if unit == "ms" else 1_000_000
    ot = (df.index.astype("int64") // 1_000_000_000 * scale).to_numpy()
    out = pd.DataFrame({
        "open_time": ot, "open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(), "close": df["close"].to_numpy(), "volume": df["volume"].to_numpy(),
        "close_time": ot + 60 * scale - 1, "quote_volume": (df["volume"] * df["close"]).to_numpy(),
        "trades": 10, "taker_base": 0.0, "taker_quote": 0.0, "ignore": 0,
    })
    return out[BINANCE_COLS]


def write_zip(path: Path, rows: pd.DataFrame, header: bool = False) -> Path:
    """Write rows as a single CSV inside a zip, the way Binance ships them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = rows.to_csv(index=False, header=header)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(path.name.replace(".zip", ".csv"), csv)
    return path


def write_months(raw: Path, symbol: str, minutes: pd.DataFrame) -> list[Path]:
    """Split minutes into monthly zips; months from 2025 on use microseconds like Binance."""
    paths = []
    for period, part in minutes.groupby(minutes.index.strftime("%Y-%m")):
        unit = "us" if int(period[:4]) >= 2025 else "ms"
        paths.append(write_zip(raw / f"{symbol}-1m-{period}.zip", to_binance(part, unit)))
    return paths
