"""Walk-forward evaluation and the one-shot final holdout.

Walk-forward: train -> calibrate -> decide -> backtest over every fold, compared with
buy-and-hold and random entries. Writes reports/walk_forward.md.

Holdout: fits the final model on all development data (calibrator and thresholds from the
walk-forward out-of-fold predictions), saves it for live use, evaluates it on the holdout
period ONCE and writes reports/holdout.md. A second run is refused unless --i-understand is
passed, because looking at the holdout twice turns it into a validation set.

    python -m src.evaluate [--model xgb] [--no-retrain]
    python -m src.evaluate --model cnn --no-retrain     # after `python -m src.cnn`
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
    if oof is None and retrain and kind == "cnn":
        from src.cnn import run as run_cnn

        oof = run_cnn(cfg)
    elif oof is None:
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


# ---------------------------------------------------------------- final model and holdout

def final_bundle_path(cfg: Config, kind: str, symbol: str):
    return resolve_path(cfg, "models") / f"final_{kind}_{symbol}.pkl"


def fit_final(cfg: Config, kind: str, symbols: list[str] | None = None) -> dict[str, dict]:
    """Fit the final model per symbol on all development data and save it with its calibrators
    (fit on all walk-forward OOF rows) and thresholds (chosen on calibrated walk-forward rows)."""
    oof = tr.load_oof(cfg, kind)
    mk = attach_market(calibrate_oof(oof, cfg), cfg)
    bundles = {}
    for sym in symbols or cfg.symbols:
        ds = tr.development_data(tr.assemble_dataset(cfg, sym), cfg)
        X, t = ds.df[ds.features].to_numpy(), ds.df.index
        fit, val = fit_val_masks(t, np.ones(len(t), dtype=bool), cfg)
        g = oof[oof["symbol"] == sym]
        v = mk[(mk["symbol"] == sym) & mk["calibrated"]]
        b = {"kind": kind, "symbol": sym, "features": ds.features, "models": {}, "calibrators": {},
             "thresholds": {}, "trained_until": str(t.max()), "created": datetime.now(timezone.utc).isoformat()}
        for side in tr.SIDES:
            y = ds.df[f"{side}_label"].to_numpy()
            b["models"][side] = tr.TRAINERS[kind](X[fit], y[fit], X[val], y[val], cfg)
            b["calibrators"][side] = fit_calibrator(g[[f"{side}_{p}" for p in tr.PROBA]].to_numpy(),
                                                    g[f"{side}_label"].to_numpy(), cfg)
            pv = v[[f"{side}_{c}" for c in CAL]].to_numpy()
            thr = choose_threshold(pv[:, 2], expected_value(pv, v["atr_frac"].to_numpy(), cfg),
                                   v[f"{side}_net"].to_numpy(), cfg) if len(v) else (NO_THRESHOLD, 0.0, 0)
            b["thresholds"][side] = thr[0]
        ensure_dir(final_bundle_path(cfg, kind, sym).parent)
        joblib.dump(b, final_bundle_path(cfg, kind, sym))
        bundles[sym] = b
        print(f"final {kind} {sym}: trained until {b['trained_until']}, thresholds {b['thresholds']}")
    return bundles


def load_bundle(cfg: Config, kind: str, symbol: str) -> dict:
    return joblib.load(final_bundle_path(cfg, kind, symbol))


def bundle_decide(bundle: dict, feats: pd.DataFrame, atr_frac: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Calibrated probabilities and decisions for feature rows using a final bundle."""
    X = feats[bundle["features"]].to_numpy()
    out = pd.DataFrame(index=feats.index)
    probs = {}
    for side in tr.SIDES:
        raw = tr.predict_proba(bundle["models"][side], X)
        probs[side] = bundle["calibrators"][side].transform(raw)
        for j, c in enumerate(CAL):
            out[f"{side}_{c}"] = probs[side][:, j]
    dec, ev_l, ev_s = decide(probs["long"], probs["short"], atr_frac,
                             bundle["thresholds"]["long"], bundle["thresholds"]["short"], cfg)
    out["decision"] = dec
    out["ev"] = np.where(dec == LONG, ev_l, np.where(dec == SHORT, ev_s, 0.0))
    out["confidence"] = np.where(dec == LONG, probs["long"][:, 2],
                                 np.where(dec == SHORT, probs["short"][:, 2], np.nan))
    return out


def holdout_lock(cfg: Config):
    return resolve_path(cfg, "reports") / "holdout.lock"


def run_holdout(cfg: Config, kind: str, i_understand: bool = False) -> dict:
    if kind not in tr.TRAINERS:
        raise SystemExit(f"The holdout supports the tabular models {sorted(tr.TRAINERS)}, not {kind!r}.")
    lock = holdout_lock(cfg)
    if lock.exists() and not i_understand:
        raise SystemExit(f"The holdout was already evaluated ({lock.read_text().strip()}).\n"
                         "Running it again would turn it into a validation set. "
                         "Pass --i-understand to run anyway; say so when reporting results.")
    bundles = fit_final(cfg, kind)
    parts = []
    for sym, b in bundles.items():
        ds = tr.assemble_dataset(cfg, sym)
        df = ds.df[holdout_mask(ds.df.index, cfg)]
        if df.empty:
            continue
        d = bundle_decide(b, df, (df["atr"] / df["close"]).to_numpy(), cfg)
        part = df.join(d).reset_index()
        part["symbol"] = sym
        part["fold"] = 0
        parts.append(part)
    if not parts:
        raise SystemExit("No labelled holdout rows yet.")
    dec = pd.concat(parts, ignore_index=True)
    ev = evaluate_decisions(dec, cfg, f"{kind} holdout")
    ev["cfg"] = cfg
    reports = ensure_dir(resolve_path(cfg, "reports"))
    figs = ensure_dir(reports / "figures")
    plot_equity({kind: ev["result"].equity, "buy & hold": ev["buy_hold_eq"]}, figs / "equity_holdout.png",
                f"Holdout {kind} (after costs)")
    save_trades(ev["result"], reports / "trades_holdout.csv")
    thr = "; ".join(f"{s}: long {b['thresholds']['long']:.2f}, short {b['thresholds']['short']:.2f}"
                    for s, b in bundles.items()).replace("inf", "off")
    md = [f"# Final holdout: {kind}", "",
          f"Evaluated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC on {ev['start']:%Y-%m-%d} to "
          f"{ev['end']:%Y-%m-%d}" + (" (**re-run with --i-understand**)" if lock.exists() else " (first run)") + ".",
          "Model fit on all development data; calibrators and thresholds from walk-forward OOF only.", "",
          "## Setup", "", *setup_lines(cfg), f"- Thresholds: {thr}", "",
          "## Verdict", "", verdict(ev), "",
          "## Results", "",
          metrics_table({kind: ev["model"], "buy & hold": ev["buy_hold"], "random entries (median)": ev["random"]}),
          "", f"![equity](figures/equity_holdout.png)", "",
          f"Signals: {int((dec['decision'] == LONG).sum())} long, {int((dec['decision'] == SHORT).sum())} short, "
          f"{int((dec['decision'] == FLAT).sum())} no trade."]
    (reports / "holdout.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    lock.write_text(json.dumps({"kind": kind, "at": datetime.now(timezone.utc).isoformat()}))
    print(f"holdout {kind}: {ev['model']['trades']} trades, net {_pct(ev['model']['net_return'])}. "
          f"Saved {reports / 'holdout.md'}")
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="xgb", choices=sorted(tr.TRAINERS) + ["cnn"])
    p.add_argument("--no-retrain", action="store_true", help="reuse the saved OOF predictions")
    p.add_argument("--holdout", action="store_true", help="evaluate the final model on the holdout (once)")
    p.add_argument("--i-understand", action="store_true", help="allow a second holdout run")
    a = p.parse_args()
    cfg = load_config()
    if a.holdout:
        run_holdout(cfg, a.model, a.i_understand)
    else:
        run_walk_forward(cfg, a.model, retrain=not a.no_retrain)


if __name__ == "__main__":
    main()
