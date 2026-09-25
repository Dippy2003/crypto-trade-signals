from src.config import ROOT, load_config, resolve_path


def test_config_loads_with_required_sections():
    cfg = load_config()
    for section in ["symbols", "paths", "labels", "features", "splits", "costs", "backtest"]:
        assert section in cfg
    assert cfg.labels.tp_atr > cfg.labels.sl_atr > 0
    assert cfg.costs.fee_per_side == 0.001


def test_overrides_merge_and_paths_resolve(tmp_path):
    cfg = load_config(overrides={"labels": {"horizon_h": 6}, "paths": {"raw": str(tmp_path)}})
    assert cfg.labels.horizon_h == 6
    assert cfg.labels.tp_atr == 2.0          # untouched keys survive the merge
    assert resolve_path(cfg, "raw") == tmp_path
    assert resolve_path(load_config(), "raw") == ROOT / "data" / "raw"
