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


def walk_forward_markdown(ev: dict, folds: pd.DataFrame, thresholds: pd.DataFrame, cal_rel: str,
                          kind: str, fig_name: str, n_warmup: int) -> str:
    res = ev["result"]
    lines = [f"# Walk-forward evaluation: {kind}", "",
             f"Traded period: {ev['start']:%Y-%m-%d} to {ev['end']:%Y-%m-%d}. "
             f"{n_warmup:,} out-of-fold rows in warm-up folds (no calibration history) are not traded.", "",
             "## Setup", "", *setup_lines(ev["cfg"]), "",
             "## Verdict", "", verdict(ev), "",
             "## Results", "",
             metrics_table({kind: ev["model"], "buy & hold": ev["buy_hold"],
                            "random entries (median)": ev["random"]}), "",
             f"Random entries: {len(ev['random_rets'])} simulations with the same trade count; "
             f"net return 5-95% range {_pct(ev['random'].get('net_return_p05'))} to "
             f"{_pct(ev['random'].get('net_return_p95'))}. Skipped signals: {res.skipped}.", "",
             f"![equity](figures/{fig_name})", "",
             "## Per fold", "",
             "| Fold | From | To | Longs | Shorts | Trades | Net return | Win rate | Buy & hold |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in folds.to_dict("records"):
        wr = "n/a" if np.isnan(r["win_rate"]) else f"{r['win_rate']:.1%}"
        lines.append(f"| {r['fold']} | {r['from']} | {r['to']} | {r['longs']} | {r['shorts']} | {r['trades']} | "
                     f"{_pct(r['net_return'])} | {wr} | {_pct(r['buy_hold'])} |")
    lines += ["", "## Thresholds chosen on earlier folds", "",
              "| Symbol | Fold | Validation rows | Long thr | Short thr | Val profit long | Val profit short |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in thresholds.to_dict("records"):
        fmt = lambda x: "off" if x == NO_THRESHOLD else f"{x:.2f}"
        lines.append(f"| {r['symbol']} | {r['fold']} | {r['val_rows']:,} | {fmt(r['thr_long'])} | "
                     f"{fmt(r['thr_short'])} | {r['val_profit_long']:+.3f} | {r['val_profit_short']:+.3f} |")
    lines += ["", "## Calibration (calibrated P(WIN) vs observed WIN rate)", "", cal_rel, ""]
    if not res.trades.empty:
        lines += ["## By symbol and side", "", "| Symbol | Side | Trades | Win rate | Mean net return |",
                  "|---|---|---:|---:|---:|"]
        for (s, sd), g in res.trades.groupby(["symbol", "side"]):
            lines.append(f"| {s} | {sd} | {len(g)} | {(g['net_ret'] > 0).mean():.1%} | {g['net_ret'].mean():+.3%} |")
    lines += ["", "Sharpe/Sortino use hourly realized equity (P/L booked at exit), annualized with sqrt(8760)."]
    return "\n".join(lines) + "\n"


def reliability_markdown(cal: pd.DataFrame) -> str:
    c = cal[cal["calibrated"]]
    parts = ["| Side | Bin mean P(WIN) | Observed WIN | Rows |", "|---|---:|---:|---:|"]
    for side in tr.SIDES:
        rel = reliability(c[f"{side}_c_win"].to_numpy(), (c[f"{side}_label"] == 2).to_numpy())
        for r in rel[rel["n"] >= 30].to_dict("records"):
            parts.append(f"| {side} | {r['predicted']:.3f} | {r['observed']:.3f} | {r['n']:,} |")
    return "\n".join(parts)


def run_walk_forward(cfg: Config, kind: str, retrain: bool = True, oof: pd.DataFrame | None = None) -> dict:
    if oof is None:
        oof = tr.run(cfg, kind) if retrain else tr.load_oof(cfg, kind)
    cal = calibrate_oof(oof, cfg)
    dec, thresholds = decide_oof(attach_market(cal, cfg), cfg)
    traded = dec[dec["calibrated"]]
    if traded.empty:
        raise SystemExit("No calibrated folds: need more walk-forward folds than calibration warm-up")
    ev = evaluate_decisions(traded, cfg, kind)
    ev["cfg"] = cfg
    reports = ensure_dir(resolve_path(cfg, "reports"))
    figs = ensure_dir(reports / "figures")
    fig = f"equity_walk_forward_{kind}.png"
    plot_equity({kind: ev["result"].equity, "buy & hold": ev["buy_hold_eq"]}, figs / fig,
                f"Walk-forward {kind} (after costs)")
    save_trades(ev["result"], reports / f"trades_{kind}.csv")
    md = walk_forward_markdown(ev, per_fold(traded, cfg), thresholds, reliability_markdown(cal), kind, fig,
                               int((~dec["calibrated"]).sum()))
    name = "walk_forward.md" if kind == "xgb" else f"walk_forward_{kind}.md"
    (reports / name).write_text(md, encoding="utf-8")
    print(f"{kind}: {ev['model']['trades']} trades, net {_pct(ev['model']['net_return'])}; "
          f"buy & hold {_pct(ev['buy_hold']['net_return'])}. Saved {reports / name}")
    ev["decisions"] = dec
    ev["thresholds"] = thresholds
    return ev
