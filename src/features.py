"""Scale-free features on closed hourly bars.

The row for bar T (open time T, closes at T + 1h) only uses bars with open time <= T, so
it is known at decision time T + 1h. No raw prices: only returns, ratios and oscillators.
Other symbols get the context symbol's (BTC) features as extra ``btc_*`` columns.

Output: data/processed/{SYM}_features.parquet

    python -m src.features
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import Config, ensure_dir, load_config, resolve_path
from src.data_loader import load_hourly
from src.labels import load_labels, wilder_atr

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_features(h: pd.DataFrame, fc: Config) -> pd.DataFrame:
    """Features for one symbol. ``fc`` is the ``features`` config section."""
    # Work on a complete hourly grid so that shift(n) always means n hours; dropped hours are NaN.
    grid = pd.date_range(h.index[0], h.index[-1], freq="1h", name=h.index.name)
    g = h.reindex(grid)
    o, hi, lo, c, v = (g[k] for k in ["open", "high", "low", "close", "volume"])
    f: dict[str, pd.Series] = {}

    for n in fc.return_windows:
        f[f"ret_{n}h"] = c / c.shift(n) - 1

    spans = sorted(fc.ema_spans)
    ema = {s: _ema(c, s) for s in spans}
    for s in spans:
        f[f"close_ema{s}"] = c / ema[s] - 1
    for a, b in zip(spans[:-1], spans[1:]):
        f[f"ema{a}_ema{b}"] = ema[a] / ema[b] - 1

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / fc.rsi_n, adjust=False, min_periods=fc.rsi_n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / fc.rsi_n, adjust=False, min_periods=fc.rsi_n).mean()
    f[f"rsi{fc.rsi_n}"] = up / (up + dn)                     # RSI / 100, defined when up + dn > 0

    macd = (_ema(c, fc.macd_fast) - _ema(c, fc.macd_slow)) / c
    signal = macd.ewm(span=fc.macd_signal, adjust=False, min_periods=fc.macd_signal).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd, signal, macd - signal

    mid = c.rolling(fc.bb_n).mean()
    sd = c.rolling(fc.bb_n).std(ddof=0)
    upper, lower = mid + fc.bb_k * sd, mid - fc.bb_k * sd
    f["bb_pctb"] = (c - lower) / (upper - lower)
    f["bb_width"] = (upper - lower) / mid

    f[f"atr{fc.atr_n}_close"] = wilder_atr(h, fc.atr_n).reindex(grid) / c

    logret = np.log(c).diff()
    for n in fc.vol_windows:
        f[f"rv_{n}h"] = logret.rolling(n).std()

    f["hl_range"] = (hi - lo) / c
    f["co_range"] = c / o - 1
    f[f"vol_rel{fc.volume_mean_n}"] = v / v.rolling(fc.volume_mean_n).mean()

    hour, dow = grid.hour.to_numpy(), grid.dayofweek.to_numpy()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    out = pd.DataFrame({k: np.asarray(s, dtype="float64") for k, s in f.items()}, index=grid)
    return out.reindex(h.index).replace([np.inf, -np.inf], np.nan)


def context_prefix(symbol: str) -> str:
    return symbol.removesuffix("USDT").lower() + "_"


def add_context(feats: pd.DataFrame, ctx: pd.DataFrame, ctx_symbol: str) -> pd.DataFrame:
    """Join the context symbol's features (without time features) onto ``feats`` by bar time."""
    cols = [c for c in ctx.columns if c not in TIME_FEATURES]
    return feats.join(ctx[cols].add_prefix(context_prefix(ctx_symbol)), how="left")


def build_features(hourly: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    """Features for every symbol in ``hourly``, with context columns for non-context symbols."""
    base = {s: compute_features(h, cfg.features) for s, h in hourly.items()}
    ctx = cfg.context_symbol
    out = {}
    for sym, f in base.items():
        if sym != ctx and ctx in base:
            f = add_context(f, base[ctx], ctx)
        out[sym] = f
    return out


def label_correlations(feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of every feature with each side's label and realized return."""
    df = feats.join(labels, how="inner")
    targets = [f"{s}_{k}" for s in ("long", "short") for k in ("label", "ret")]
    return pd.DataFrame({t: df[feats.columns].corrwith(df[t]) for t in targets})


def leakage_flags(corr: pd.DataFrame, max_abs_corr: float) -> list[str]:
    """Features whose |corr| with any target exceeds the limit (to be investigated, not trusted)."""
    return sorted(corr.index[(corr.abs() > max_abs_corr).any(axis=1)])


def leakage_report(cfg: Config, symbols: list[str]) -> str:
    lim = cfg.features.leakage_max_abs_corr
    lines = ["# Feature leakage check", "",
             f"Correlation of each feature with labels and realized returns. |corr| > {lim} is flagged.", ""]
    for sym in symbols:
        corr = label_correlations(load_features(cfg, sym), load_labels(cfg, sym))
        flags = leakage_flags(corr, lim)
        top = corr.abs().max(axis=1).sort_values(ascending=False).head(10)
        lines += [f"## {sym}", "", f"Flagged: {', '.join(flags) if flags else 'none'}", "",
                  "| Feature | max |corr| |", "|---|---:|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in top.items()]
        lines.append("")
    return "\n".join(lines)


def load_features(cfg: Config, symbol: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_path(cfg, "processed") / f"{symbol}_features.parquet")


def build(cfg: Config, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    proc = resolve_path(cfg, "processed")
    symbols = symbols or [p.name.split("_")[0] for p in sorted(proc.glob("*_1h.parquet"))]
    need = set(symbols) | ({cfg.context_symbol} if (proc / f"{cfg.context_symbol}_1h.parquet").exists() else set())
    feats = build_features({s: load_hourly(cfg, s) for s in sorted(need)}, cfg)
    for sym in symbols:
        f = feats[sym]
        f.to_parquet(proc / f"{sym}_features.parquet")
        print(f"{sym}: {f.shape[1]} features, {f.dropna().shape[0]:,} of {len(f):,} rows complete")
    return {s: feats[s] for s in symbols}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--check-leakage", action="store_true",
                   help="also write reports/leakage.md (needs labels from src.labels)")
    a = p.parse_args()
    cfg = load_config()
    feats = build(cfg, a.symbols)
    if a.check_leakage:
        path = ensure_dir(resolve_path(cfg, "reports")) / "leakage.md"
        path.write_text(leakage_report(cfg, list(feats)), encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
