"""Walk-forward evaluation and the one-shot final holdout.

Walk-forward: train -> calibrate -> decide -> backtest over every fold, compared with
buy-and-hold and random entries. Writes reports/walk_forward.md.

Holdout: fits the final model on all development data (calibrator and thresholds from the
walk-forward out-of-fold predictions), saves it for live use, evaluates it on the holdout
period ONCE and writes reports/holdout.md. A second run is refused unless --i-understand is
passed, because looking at the holdout twice turns it into a validation set.

    python -m src.evaluate [--model xgb] [--no-retrain]
    python -m src.evaluate --holdout [--model xgb]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src import train as tr
from src.backtest import backtest, save_trades
from src.calibrate import CAL, calibrate_oof, fit_calibrator, reliability
from src.config import Config, ensure_dir, load_config, resolve_path
from src.data_loader import load_hourly
from src.decision import (FLAT, LONG, NO_THRESHOLD, SHORT, attach_market, choose_threshold, decide,
                          decide_oof, expected_value, round_trip_cost)
from src.metrics import buy_and_hold, metrics_table, plot_equity, random_baseline, summarize
from src.splits import fit_val_masks, holdout_mask

H1 = pd.Timedelta(hours=1)


def closes(cfg: Config, symbols) -> dict[str, pd.Series]:
    return {s: load_hourly(cfg, s)["close"] for s in symbols}


def evaluate_decisions(dec: pd.DataFrame, cfg: Config, name: str) -> dict:
    """Backtest decided rows and compute the model vs baseline metrics over their period."""
    start = dec["time"].min() + H1
    end = dec["time"].max() + H1 + pd.Timedelta(hours=cfg.labels.horizon_h)
    res = backtest(dec, cfg, start, end)
    model = summarize(res, cfg)
    bh, bh_eq = buy_and_hold(closes(cfg, dec["symbol"].unique()), start, end, cfg)
    rnd, rnd_rets = random_baseline(dec, len(res.trades), cfg, start, end)
    return {"name": name, "result": res, "model": model, "buy_hold": bh, "buy_hold_eq": bh_eq,
            "random": rnd, "random_rets": rnd_rets, "start": start, "end": end}


def per_fold(dec: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for k, g in dec.groupby("fold"):
        start, end = g["time"].min() + H1, g["time"].max() + H1 + pd.Timedelta(hours=cfg.labels.horizon_h)
        res = backtest(g, cfg, start, end)
        s = summarize(res, cfg)
        bh, _ = buy_and_hold(closes(cfg, g["symbol"].unique()), start, end, cfg)
        rows.append({"fold": k, "from": start.date(), "to": (g["time"].max() + H1).date(),
                     "longs": int((g["decision"] == LONG).sum()), "shorts": int((g["decision"] == SHORT).sum()),
                     "trades": s["trades"], "net_return": s["net_return"], "win_rate": s["win_rate"],
                     "buy_hold": bh["net_return"]})
    return pd.DataFrame(rows)


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2%}"


def setup_lines(cfg: Config) -> list[str]:
    lc, c, bt = cfg.labels, cfg.costs, cfg.backtest
    return [f"- Barriers: TP {lc.tp_atr} x ATR({lc.atr_n}), SL {lc.sl_atr} x ATR, horizon {lc.horizon_h}h, "
            "entry at the first minute after the signal bar",
            f"- Costs: fee {c.fee_per_side:.2%} per side + slippage {c.slippage_per_side:.2%} per side "
            f"(round trip {round_trip_cost(cfg):.2%})",
            f"- Sizing: risk {bt.risk_per_trade:.0%} of equity per trade, max leverage {bt.max_position_leverage}x, "
            f"daily loss stop {bt.max_daily_loss:.0%}",
            f"- Walk-forward: {cfg.splits.test_months}-month test periods from {cfg.splits.first_test_start}, "
            f"embargo {cfg.splits.embargo_h}h, holdout from {cfg.splits.holdout_start} untouched"]


def verdict(ev: dict) -> str:
    m, rnd, bh = ev["model"], ev["random"], ev["buy_hold"]
    if m["trades"] == 0:
        return ("**No trades.** On every validation window no probability threshold produced a profit "
                "after costs, so the decision rule stayed flat. This is a valid result: the model shows no "
                "tradable edge after costs under this setup.")
    p95 = rnd.get("net_return_p95", np.nan)
    beats_random = m["net_return"] > p95
    txt = (f"The model made {m['trades']} trades for {_pct(m['net_return'])} vs buy-and-hold "
           f"{_pct(bh['net_return'])} and a random-entry median of {_pct(rnd['net_return'])} "
           f"(95th percentile {_pct(p95)}). ")
    if m["net_return"] <= 0:
        return txt + "**No edge after costs.**"
    return txt + ("It beats 95% of random-entry runs." if beats_random else
                  "**It does not beat the random-entry range, so the result is not distinguishable from luck.**")
