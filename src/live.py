"""Live signal for the latest closed hourly candle.

Fetches closed 1h candles from the Binance public REST API (/api/v3/klines, no key), builds
features with the same code as training (including BTC context for ETH), applies the final
model bundle saved by `python -m src.evaluate --holdout` (model + calibrators + thresholds)
and prints LONG / SHORT / NO TRADE with entry, take-profit, stop-loss and confidence.

The entry shown is the last close; the real entry is the open of the next minute.
Research only, not financial advice.

    python -m src.live [--symbols BTCUSDT ETHUSDT] [--model xgb]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from src.config import Config, load_config
from src.decision import LONG, NAMES, SHORT
from src.evaluate import bundle_decide, load_bundle
from src.features import build_features
from src.labels import wilder_atr

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def _get(session, url: str, params: dict, timeout: float) -> list:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def klines_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype("float64")
    df["time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = df["close_time"].astype("int64")
    return df.set_index("time")
