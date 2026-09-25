# reports/

Generated output lands here and is not committed. Each command writes its own files:

| File | Written by |
|---|---|
| `data_quality.md` | `python -m src.data_loader` |
| `label_distribution.md` | `python -m src.labels` |
| `leakage.md` | `python -m src.features --check-leakage` |
| `train_<model>.md`, `figures/` | `python -m src.train` |
| `walk_forward.md`, `trades_<model>.csv` | `python -m src.evaluate` |
| `holdout.md` | `python -m src.evaluate --holdout` (runs once) |
| `comparison.md` | `python -m src.compare` |
| `paper_trades.csv`, `paper_summary.md` | `python -m src.paper_trade` |
