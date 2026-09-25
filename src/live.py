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


def fetch_hourly(symbol: str, cfg: Config, bars: int | None = None, session=None,
                 now: pd.Timestamp | None = None) -> pd.DataFrame:
    """The most recent ``bars`` CLOSED hourly candles (paginating backwards)."""
    lv = cfg.live
    session = session or requests.Session()
    bars = bars or lv.history_bars
    now_ms = int((now or pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    parts, end = [], now_ms
    while sum(len(p) for p in parts) < bars + 1:
        rows = _get(session, lv.api_url, {"symbol": symbol, "interval": "1h", "limit": lv.request_limit,
                                          "endTime": end}, lv.timeout_s)
        if not rows:
            break
        df = klines_frame(rows)
        parts.insert(0, df)
        end = int(df["open_time"].iloc[0]) - 1
        if len(rows) < lv.request_limit:
            break
    h = pd.concat(parts)
    h = h[~h.index.duplicated()].sort_index()
    h = h[h["close_time"] < now_ms]                       # drop the candle that is still forming
    return h[["open", "high", "low", "close", "volume"]].tail(bars)


def fetch_minutes(symbol: str, cfg: Config, start: pd.Timestamp, end: pd.Timestamp, session=None) -> pd.DataFrame:
    """Closed 1m candles with open time in [start, end) (paginating forwards)."""
    lv = cfg.live
    session = session or requests.Session()
    parts, cur = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        rows = _get(session, lv.api_url, {"symbol": symbol, "interval": "1m", "limit": lv.request_limit,
                                          "startTime": cur, "endTime": end_ms - 1}, lv.timeout_s)
        if not rows:
            break
        df = klines_frame(rows)
        parts.append(df)
        cur = int(df["open_time"].iloc[-1]) + 60_000
        if len(rows) < lv.request_limit:
            break
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    m = pd.concat(parts)
    m = m[~m.index.duplicated()].sort_index()
    return m[["open", "high", "low", "close", "volume"]]


@dataclass
class Signal:
    symbol: str
    bar_time: pd.Timestamp          # open time of the last closed bar
    decision_time: pd.Timestamp     # when that bar closed
    decision: str                   # LONG / SHORT / NO TRADE
    entry: float                    # last close (estimate of the next minute's open)
    tp: float | None
    sl: float | None
    confidence: float | None        # calibrated P(WIN) of the chosen side
    p_long_win: float
    p_short_win: float
    ev: float
    atr: float
    model: str


def make_signal(cfg: Config, symbol: str, hourly: dict[str, pd.DataFrame], bundle: dict) -> Signal:
    """Signal for the last bar of ``hourly[symbol]`` (other entries give context features)."""
    feats = build_features(hourly, cfg)[symbol]
    row = feats.iloc[[-1]]
    missing = [c for c in bundle["features"] if pd.isna(row[c].iloc[0])]
    if missing:
        raise ValueError(f"{symbol}: features not ready for the last bar ({missing[:5]}...); need more history")
    h = hourly[symbol]
    atr = float(wilder_atr(h, cfg.labels.atr_n).iloc[-1])
    close = float(h["close"].iloc[-1])
    d = bundle_decide(bundle, row, np.array([atr / close]), cfg).iloc[0]
    side = int(d["decision"])
    lc = cfg.labels
    tp = sl = conf = None
    if side in (LONG, SHORT):
        tp, sl, conf = close + side * lc.tp_atr * atr, close - side * lc.sl_atr * atr, float(d["confidence"])
    t = h.index[-1]
    return Signal(symbol, t, t + pd.Timedelta(hours=1), NAMES[side], close, tp, sl, conf,
                  float(d["long_c_win"]), float(d["short_c_win"]), float(d["ev"]), atr, bundle["kind"])


def get_signal(cfg: Config, symbol: str, kind: str | None = None, session=None,
               now: pd.Timestamp | None = None) -> tuple[Signal, pd.DataFrame]:
    """Fetch data, load the final bundle and return (signal, hourly candles of ``symbol``)."""
    kind = kind or cfg.live.model_kind
    bundle = load_bundle(cfg, kind, symbol)
    need = {symbol} | ({cfg.context_symbol} if symbol != cfg.context_symbol else set())
    hourly = {s: fetch_hourly(s, cfg, session=session, now=now) for s in need}
    return make_signal(cfg, symbol, hourly, bundle), hourly[symbol]


def format_signal(s: Signal) -> str:
    head = f"{s.symbol}  bar {s.bar_time:%Y-%m-%d %H:%M} UTC (closed {s.decision_time:%H:%M})  [{s.model}]"
    if s.decision == "NO TRADE":
        body = (f"  NO TRADE   P(long WIN) {s.p_long_win:.1%}   P(short WIN) {s.p_short_win:.1%}   "
                f"last close {s.entry:,.2f}")
    else:
        body = (f"  {s.decision:<9}  entry ~{s.entry:,.2f}   TP {s.tp:,.2f}   SL {s.sl:,.2f}   "
                f"confidence {s.confidence:.1%}   EV {s.ev:+.3%}")
    return head + "\n" + body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--model")
    a = p.parse_args()
    cfg = load_config()
    for sym in a.symbols or cfg.symbols:
        try:
            sig, _ = get_signal(cfg, sym, a.model)
        except FileNotFoundError:
            print(f"{sym}: no final model; run `python -m src.evaluate --holdout` first")
            continue
        print(format_signal(sig))
    print("\nResearch only, not financial advice.")


if __name__ == "__main__":
    main()
