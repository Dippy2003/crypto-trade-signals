# crypto-trade-signals

Research project that asks, once per hour, whether a trade on **BTCUSDT** or **ETHUSDT** is worth taking:

- **LONG** with entry, take-profit, stop-loss and confidence
- **SHORT** with entry, take-profit, stop-loss and confidence
- **NO TRADE** when the setup is not good enough

> **Disclaimer.** This is a research project, not financial advice and not a trading system.
> Results are reported honestly, including "no edge after costs" if that is what the data shows.
> Do not trade real money based on anything in this repository.

## Principles

- No look-ahead: a decision at the close of hourly bar T only uses data with timestamps < T + 1h.
- Time-based validation only (walk-forward with purging and an embargo).
- Trading costs (fees and slippage) are always included.
- A final holdout period is evaluated once and never used for tuning.

## Planned pipeline

1. Download Binance spot 1-minute klines (`src/download.py`)
2. Build clean 1-minute and 1-hour candles plus a data-quality report (`src/data_loader.py`)
3. Triple-barrier labels: TP = 2×ATR, SL = 1×ATR, 12h horizon, minute-accurate (`src/labels.py`)
4. Scale-free features on closed bars (`src/features.py`)
5. Walk-forward splits with purging, embargo and a final holdout (`src/splits.py`)
6. Logistic-regression baseline and XGBoost (`src/train.py`)
7. Calibration and a cost-aware decision rule (`src/calibrate.py`, `src/decision.py`)
8. Event-driven backtest and metrics against baselines (`src/backtest.py`, `src/metrics.py`)
9. Walk-forward and holdout evaluation (`src/evaluate.py`)
10. Chart-image CNN and model comparison (`src/chart_images.py`, `src/cnn.py`, `src/compare.py`)
11. Live signals, Streamlit app and paper trading (`src/live.py`, `src/app.py`, `src/paper_trade.py`)

## Setup

```bash
python -m venv .venv            # Python 3.11
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
pytest
```

All settings live in `config/config.yaml`.
