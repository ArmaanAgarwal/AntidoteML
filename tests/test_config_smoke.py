"""Person 4 tests. The skeleton holds together and the smoke config loads."""

import torch

from antidote.system.config import load_config, pick_device
from antidote.system.pool import InProcessPool, make_pool
from antidote.ml import get_flat, make_model
from antidote.ml.data import CLASS_NAMES, load_data


def test_smoke_config_loads():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.num_workers == 5
    assert cfg.rounds == 3
    assert cfg.execution == "inprocess"
    assert cfg.attackers == {}
    assert pick_device() in ("mps", "cuda", "cpu")


def test_inprocess_pool_returns_one_update_per_active_worker():
    cfg = load_config("configs/smoke.yaml")
    splits, _, _ = load_data(cfg.dataset, cfg.num_workers, cfg.seed)
    pool = make_pool(cfg, splits, cfg.attackers)
    assert isinstance(pool, InProcessPool)

    global_flat = get_flat(make_model(len(CLASS_NAMES)))
    active = [0, 1, 2]
    results = pool.run_round(global_flat, 1, active)
    pool.close()

    assert sorted(results) == active
    for wid in active:
        delta = results[wid]
        assert isinstance(delta, torch.Tensor)
        assert delta.dtype == torch.float32
        assert delta.ndim == 1
        assert delta.shape == global_flat.shape
        assert torch.isfinite(delta).all()


def test_model_has_no_buffers():
    model = make_model(len(CLASS_NAMES))
    assert list(model.buffers()) == []
