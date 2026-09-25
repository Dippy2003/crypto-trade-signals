# crypto-trade-signals

Research project that asks, once per hour, whether a trade on **BTCUSDT** or **ETHUSDT** is worth taking:

- **LONG** with entry, take-profit, stop-loss and confidence
- **SHORT** with entry, take-profit, stop-loss and confidence
- **NO TRADE** when the setup is not good enough

> **Not financial advice.** This is a research project, not a trading system. It is built to find out
> *whether* there is an edge after costs, and it reports "no edge" when that is what the data shows.
> Do not trade real money based on anything in this repository.

## Principles

- **No look-ahead.** A decision at the close of hourly bar T only uses data with timestamps < T + 1h.
  Features are computed on closed bars and tested for this (`tests/test_leakage.py`).
- **Time-based validation only.** Expanding walk-forward folds with purging and an embargo.
- **Costs always included.** 0.1% fee per side plus slippage on both fills.
- **One-shot holdout.** 2026-01-01 onward is evaluated once, after everything else is fixed.

## Setup

Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows (use `source .venv/bin/activate` on Linux/macOS)
pip install -r requirements.txt
pytest                              # ~1 minute, synthetic data only
```

`requirements.txt` installs the CPU build of PyTorch. For an NVIDIA GPU (used automatically by the CNN):

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
```

All settings (symbols, barriers, fees, split dates, thresholds, model parameters) are in
`config/config.yaml`. Set `CTS_CONFIG` to use a different file.

## Pipeline, in order

| # | Command | Output |
|---|---|---|
| 1 | `python -m src.download` | `data/raw/*.zip` (+ `.CHECKSUM`), monthly from 2021-01 plus daily for the current month |
| 2 | `python -m src.data_loader` | `data/processed/{SYM}_1m.parquet`, `{SYM}_1h.parquet`, `reports/data_quality.md` |
| 3 | `python -m src.labels` | `data/processed/{SYM}_labels.parquet`, `reports/label_distribution.md` |
| 4 | `python -m src.features --check-leakage` | `data/processed/{SYM}_features.parquet`, `reports/leakage.md` |
| 5 | `python -m src.splits` | prints the walk-forward fold table |
| 6 | `python -m src.evaluate --model logreg` | trains the baseline; `reports/train_logreg.md`, `reports/walk_forward_logreg.md` |
| 7 | `python -m src.evaluate --model xgb` | trains XGBoost; `reports/train_xgb.md`, **`reports/walk_forward.md`** |
| 8 | `python -m src.chart_images` | `data/images/{SYM}/YYYYMMDD_HH.png` (resumable) |
| 9 | `python -m src.evaluate --model cnn` | trains the CNN; `reports/train_cnn.md`, `reports/walk_forward_cnn.md` |
| 10 | `python -m src.compare` | **`reports/comparison.md`** |
| 11 | `python -m src.evaluate --holdout` | final model in `models/`, **`reports/holdout.md`**; runs once |
| 12 | `python -m src.live` / `streamlit run src/app.py` | current signal / web app |
| 13 | `python -m src.paper_trade --loop`, then `--summary` | `reports/paper_trades.csv`, `reports/paper_summary.md` |

Steps 6, 7 and 9 retrain; add `--no-retrain` to re-evaluate saved out-of-fold predictions.
`python -m src.train --model xgb` and `python -m src.cnn` run the training step on its own
(`python -m src.cnn --folds 5 6 --epochs 3` for a quick look).
Run step 11 only when the walk-forward setup is final: a second run is refused unless you pass
`--i-understand`, and any number from a second run should be reported as such.

## Method

**Data.** Binance spot 1-minute klines. Timestamps are milliseconds before 2025 and microseconds from
2025; both are converted to UTC. Rows are deduplicated and sorted. Hourly bars with fewer than 50 one-minute
bars are dropped. `reports/data_quality.md` lists missing minutes, the largest gap and dropped hours.

**Labels (triple barrier).** For hourly bar T the decision is made when the bar closes (T + 1h) and the
entry is the open of that next minute. LONG: TP = entry + 2×ATR(14), SL = entry − 1×ATR(14); SHORT is
mirrored. The next 12 hours are checked minute by minute: TP first = WIN, SL first = LOSS, both in the same
minute = LOSS (conservative), neither = NO TRADE (exit at the last close). Windows with a missing minute are
skipped. The vectorized labeler is tested bit-for-bit against the original loop.

**Features.** Scale-free only: returns over 1–24h, close/EMA and EMA/EMA ratios (10/20/50/200), RSI14, MACD
(normalized by price), Bollinger %B and width, ATR/close, realized volatility (6/12/24h), bar ranges, volume
relative to its 20-bar mean, hour and weekday as sin/cos. ETH also gets BTC's features as `btc_*` columns.
Leakage tests rebuild features on data truncated at T + 1h and require row T to be identical, and flag any
feature with |corr| > 0.3 against a label or realized return.

**Walk-forward splits.** Quarterly test periods from 2023-01-01 to the holdout; each fold trains on
everything before its test period. A row's information spans [T, T + 1h + 12h]; training rows whose span comes
within the 24h embargo of the test period are **purged**. XGBoost and the CNN early-stop on a purged
validation tail at the end of each training fold. Rows touching the holdout are removed from all development
work.

**Models.** Per symbol and side, 3-class (LOSS / NO TRADE / WIN): logistic regression (standardized on the
training fold only, balanced class weights), XGBoost (balanced weights, early stopping) and a ResNet18 CNN
(ImageNet weights, new 3-class head) on 224×224 images of the last 60 hourly candles in one fixed style.

**Calibration.** One-vs-rest isotonic regression (or Platt) on out-of-fold predictions. Fold k is calibrated
only with OOF rows from earlier folds whose labels were known before fold k started; the first fold is warm-up.

**Decision rule.** LONG if calibrated P(long WIN) ≥ threshold, its expected value after costs is positive, and
it beats SHORT; mirrored for SHORT; otherwise NO TRADE.
EV = P(WIN)·2·ATR/price − P(LOSS)·1·ATR/price − round-trip costs. Thresholds are chosen per symbol and side on
earlier calibrated folds by realized net profit; a side with no profitable threshold is switched off.

**Backtest.** Event-driven on the minute-accurate label outcomes: one position per symbol, fees and slippage
on both fills, 1% of equity at risk per trade (position size from the stop distance, capped at 2× leverage)
and no new trades for the rest of a UTC day after a 3% realized loss.

**Metrics and baselines.** Net return, trades, win rate, average win/loss, profit factor, max drawdown,
exposure, and Sharpe/Sortino from hourly equity returns annualized with √8760. Every result is shown next to
buy-and-hold and random entries (same trade count, random time and side, same TP/SL, sizing and costs, 200
simulations). `compare` adds bootstrap confidence intervals on the difference in mean trade return.

## Results

Fill these tables from the reports after running the pipeline above.

**Walk-forward** (`reports/walk_forward.md`, `reports/comparison.md`)

| Strategy | Net return | Trades | Win rate | Profit factor | Max DD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | | | | | | |
| CNN (chart images) | | | | | | |
| Logistic regression | | | | | | |
| Random entries (median) | | | | | | |
| Buy & hold | | | | | | |

**Holdout** (`reports/holdout.md`, evaluated once)

| Strategy | Net return | Trades | Win rate | Profit factor | Max DD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (final) | | | | | | |
| Buy & hold | | | | | | |

**Preliminary run during development** (BTCUSDT only, data 2023-01 to 2026-08, before the full download):

- Labels: about 26% WIN, 17% NO TRADE and 57% LOSS on each side; with a 2:1 target a trade needs more than
  33% wins before costs.
- The largest |corr| between any feature and a label or return was 0.04 (limit 0.3).
- XGBoost walk-forward over 2023-10 to 2025-12: **0 trades**. With ATR at about 0.63% of price, the 0.24%
  round-trip cost is about 38% of the stop distance, and the calibrated model almost never predicted
  positive EV after costs. On no validation window did any threshold make money, so the rule stayed flat.
  Buy-and-hold returned +226% over the same period.

This "no edge after costs" result is the honest starting point. The pipeline still reports every result.
For example, the random baseline and bootstrap intervals are there to show whether a change beats chance.

## Limitations

- Fees and slippage are fixed; real slippage grows with volatility and size, and funding or borrow costs
  for shorts are ignored.
- Barrier outcomes use 1-minute highs and lows; the order of events inside a minute is unknown (handled
  conservatively: both barriers in one minute = LOSS).
- Sharpe and Sortino use realized equity (P/L booked at exit), which understates intra-trade volatility.
- Trades overlap in time, so the bootstrap intervals are optimistic.
- Threshold search uses the sum of overlapping validation trades, not a full backtest.
- Only two symbols and one bar size; regimes change and the past is a weak guide.
- The final model is trained on data up to the holdout start; retrain with more data before relying on
  live signals, and treat paper-trading results with fewer than a few hundred trades as anecdotal.
- Live features warm up on 2,000 hourly bars, so EMA200 values differ from training by about 1e-6.

## Project structure

```
config/config.yaml        all settings
src/config.py             settings loader
src/download.py           Binance data download with checksums
src/data_loader.py        1m/1h candles and data-quality report
src/labels.py             triple-barrier labels
src/features.py           features and leakage check
src/splits.py             walk-forward folds, purging, embargo, holdout
src/train.py              dataset, logistic regression, XGBoost
src/calibrate.py          probability calibration
src/decision.py           cost-aware decision rule and thresholds
src/backtest.py           event-driven backtest
src/metrics.py            metrics, plots, baselines
src/evaluate.py           walk-forward and holdout evaluation
src/chart_images.py       candlestick images for the CNN
src/cnn.py                ResNet18 on chart images
src/compare.py            model comparison with bootstrap CIs
src/live.py               live signal from the Binance REST API
src/app.py                Streamlit app
src/paper_trade.py        paper trading log and summary
tests/                    pytest suite on synthetic data
```

## Disclaimer

This software is for research and education. It is **not financial advice**, makes no promise of profit,
and past or simulated performance does not predict future results. Crypto trading can lose all of the money
put into it.
