"""Compare XGBoost (numbers), CNN (chart images), logistic regression and random entries.

All models are evaluated on the same (symbol, time) rows, the same folds, the same
calibration warm-up, the same decision rule, costs and backtester. Significance: a bootstrap
confidence interval on the difference in mean net trade return between each pair.

Output: reports/comparison.md

    python -m src.compare [--models xgb cnn logreg]
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from src.backtest import backtest
from src.calibrate import calibrate_oof
from src.config import Config, ensure_dir, load_config, resolve_path
from src.decision import LONG, SHORT, attach_market, decide_oof
from src.evaluate import H1, setup_lines
from src.metrics import metrics_table, plot_equity, summarize
from src.train import load_oof, oof_path

DEFAULT_MODELS = ["xgb", "cnn", "logreg"]
KEY = ["symbol", "time"]


def load_available(cfg: Config, kinds: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for k in kinds:
        if oof_path(cfg, k).exists():
            out[k] = load_oof(cfg, k)
        else:
            print(f"{k}: no out-of-fold predictions ({oof_path(cfg, k).name}); skipped")
    return out


def common_rows(oofs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every model to the (symbol, time) rows all of them predicted."""
    keys = None
    for o in oofs.values():
        k = pd.MultiIndex.from_frame(o[KEY])
        keys = k if keys is None else keys.intersection(k)
    out = {}
    for name, o in oofs.items():
        keep = pd.MultiIndex.from_frame(o[KEY]).isin(keys)
        out[name] = o[keep].sort_values(KEY).reset_index(drop=True)
    return out


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, level: float, seed: int) -> tuple[float, float, float]:
    """Difference in means (a - b) with a percentile bootstrap CI, resampling each side independently."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def random_trades(pool: pd.DataFrame, n: int, cfg: Config, start, end) -> tuple[pd.DataFrame, list[dict]]:
    """Trades from ``random_sims`` random-entry runs (same count, random side) and their summaries."""
    rng = np.random.default_rng(cfg.model.seed)
    trades, sums = [], []
    for _ in range(cfg.metrics.random_sims if n else 0):
        pick = pool.iloc[np.sort(rng.choice(len(pool), min(n, len(pool)), replace=False))].copy()
        pick["decision"] = rng.choice([LONG, SHORT], len(pick))
        res = backtest(pick, cfg, start, end)
        trades.append(res.trades)
        sums.append(summarize(res, cfg))
    return (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["net_ret"])), sums


def compare(cfg: Config, kinds: list[str] | None = None) -> dict:
    oofs = load_available(cfg, kinds or DEFAULT_MODELS)
    if not oofs:
        raise SystemExit("Nothing to compare: run `python -m src.train` / `python -m src.cnn` first.")
    oofs = common_rows(oofs)
    results, decided = {}, {}
    for name, oof in oofs.items():
        dec, _ = decide_oof(attach_market(calibrate_oof(oof, cfg), cfg), cfg)
        decided[name] = dec[dec["calibrated"]]
    first = next(iter(decided.values()))
    if first.empty:
        raise SystemExit("No calibrated rows in common; need more folds.")
    start = first["time"].min() + H1
    end = first["time"].max() + H1 + pd.Timedelta(hours=cfg.labels.horizon_h)
    for name, dec in decided.items():
        res = backtest(dec, cfg, start, end)
        results[name] = {"result": res, "summary": summarize(res, cfg)}

    ref = "xgb" if "xgb" in results and results["xgb"]["summary"]["trades"] else \
        max(results, key=lambda k: results[k]["summary"]["trades"])
    n_rand = results[ref]["summary"]["trades"]
    rtr, rsums = random_trades(first, n_rand, cfg, start, end)
    if rsums:
        rs = pd.DataFrame(rsums)
        rand_summary = {c: float(rs[c].replace(np.inf, np.nan).median()) for c in rs.columns}
        rand_summary["trades"] = float(rs["trades"].mean())
    else:
        rand_summary = {"trades": 0, "net_return": 0.0}

    samples = {k: v["result"].trades["net_ret"].to_numpy() for k, v in results.items()}
    samples["random"] = rtr["net_ret"].to_numpy(dtype=float)
    mc = cfg.metrics
    tests = []
    for a, b in combinations(samples, 2):
        if len(samples[a]) < 2 or len(samples[b]) < 2:
            tests.append({"a": a, "b": b, "diff": np.nan, "lo": np.nan, "hi": np.nan,
                          "n_a": len(samples[a]), "n_b": len(samples[b])})
            continue
        d, lo, hi = bootstrap_diff(samples[a], samples[b], mc.bootstrap_n, mc.ci_level, cfg.model.seed)
        tests.append({"a": a, "b": b, "diff": d, "lo": lo, "hi": hi, "n_a": len(samples[a]), "n_b": len(samples[b])})

    out = {"results": results, "random": rand_summary, "random_ref": ref, "tests": pd.DataFrame(tests),
           "start": start, "end": end, "rows": len(first), "models": list(oofs)}
    write_report(cfg, out)
    return out


def write_report(cfg: Config, out: dict) -> None:
    reports = ensure_dir(resolve_path(cfg, "reports"))
    figs = ensure_dir(reports / "figures")
    plot_equity({k: v["result"].equity for k, v in out["results"].items()}, figs / "equity_comparison.png",
                "Model comparison (after costs)")
    level = cfg.metrics.ci_level
    rows = {k: v["summary"] for k, v in out["results"].items()}
    rows[f"random ({out['random_ref']} trade count, median)"] = out["random"]
    lines = ["# Model comparison", "",
             f"Models: {', '.join(out['models'])}. Identical rows ({out['rows']:,} decision times after "
             f"calibration warm-up), folds, decision rule, costs and backtester; traded period "
             f"{out['start']:%Y-%m-%d} to {out['end']:%Y-%m-%d}.", "",
             "## Setup", "", *setup_lines(cfg), "",
             "## Results", "", metrics_table(rows), "",
             "![equity](figures/equity_comparison.png)", "",
             f"## Difference in mean net trade return ({level:.0%} bootstrap CI, {cfg.metrics.bootstrap_n:,} resamples)", "",
             "| A | B | Trades A | Trades B | Mean A - mean B | CI low | CI high | Significant |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in out["tests"].to_dict("records"):
        if np.isnan(r["diff"]):
            lines.append(f"| {r['a']} | {r['b']} | {r['n_a']} | {r['n_b']} | n/a | n/a | n/a | too few trades |")
            continue
        sig = "yes" if r["lo"] > 0 or r["hi"] < 0 else "no"
        lines.append(f"| {r['a']} | {r['b']} | {r['n_a']:,} | {r['n_b']:,} | {r['diff']:+.4%} | "
                     f"{r['lo']:+.4%} | {r['hi']:+.4%} | {sig} |")
    lines += ["", "A difference is called significant only when its confidence interval excludes zero. "
              "Trades overlap in time and are not fully independent, so these intervals are optimistic."]
    (reports / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {reports / 'comparison.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    compare(load_config(), p.parse_args().models)


if __name__ == "__main__":
    main()
