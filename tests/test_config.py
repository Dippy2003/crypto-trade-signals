from src.config import ROOT, load_config, resolve_path


def test_config_loads_with_required_sections():
    cfg = load_config()
    for section in ["symbols", "paths", "labels", "features", "splits", "costs", "backtest"]:
        assert section in cfg
    assert cfg.labels.tp_atr > cfg.labels.sl_atr > 0
    assert cfg.costs.fee_per_side == 0.001
