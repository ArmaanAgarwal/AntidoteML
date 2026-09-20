"""Person 4 tests. The skeleton holds together and the smoke config loads."""

import torch

from antidote.ml import get_flat, make_model
from antidote.system.config import load_config, pick_device
from antidote.system.pool import InProcessPool, make_pool

NUM_CLASSES = 43  # frozen by the contract, so the tests never read ml/data.py


def fake_splits(num_workers, images=8, seed=0):
    """Synthetic worker splits, the same shape as the real ones.

    No test of mine calls load_data. A clean checkout has no GTSRB download
    sitting in front of it, and the suite runs with the network unplugged.
    """
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(images, 3, 32, 32, generator=g),
            torch.randint(0, NUM_CLASSES, (images,), generator=g, dtype=torch.int64),
        )
        for _ in range(num_workers)
    ]


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
    pool = make_pool(cfg, fake_splits(cfg.num_workers), cfg.attackers)
    assert isinstance(pool, InProcessPool)

    global_flat = get_flat(make_model(NUM_CLASSES))
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
    model = make_model(NUM_CLASSES)
    assert list(model.buffers()) == []
