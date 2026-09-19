"""Person 4 tests for MultiprocessPool.

These use smoke sized data and a short timeout so the whole file stays quick.
Real processes are started, so every test closes its pool in a finally block.
"""

import pytest
import torch

from antidote.ml import get_flat, make_model
from antidote.ml.data import CLASS_NAMES
from antidote.system.config import Config
from antidote.system.pool import InProcessPool, MultiprocessPool, make_pool

NUM_WORKERS = 3
IMAGES = 8


def make_splits(num_workers=NUM_WORKERS, n=IMAGES):
    g = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(n, 3, 32, 32, generator=g),
            torch.randint(0, len(CLASS_NAMES), (n,), generator=g),
        )
        for _ in range(num_workers)
    ]


def make_cfg(**overrides):
    return Config(
        name="unit",
        seed=7,
        num_workers=NUM_WORKERS,
        batch_size=4,
        execution="multiprocess",
        round_timeout_s=3.0,
        **overrides,
    )


@pytest.fixture
def global_flat():
    return get_flat(make_model(len(CLASS_NAMES)))


def test_make_pool_picks_the_multiprocess_pool(global_flat):
    pool = make_pool(make_cfg(), make_splits(), {})
    try:
        assert isinstance(pool, MultiprocessPool)
    finally:
        pool.close()


def test_a_round_comes_back_with_one_update_per_active_worker(global_flat):
    pool = MultiprocessPool(make_cfg(), make_splits(), {})
    try:
        results = pool.run_round(global_flat, 1, [0, 2])
        assert sorted(results) == [0, 2]
        for delta in results.values():
            assert delta is not None
            assert delta.dtype == torch.float32
            assert delta.ndim == 1
            assert delta.shape == global_flat.shape
        assert pool.status() == {0: "ok", 2: "ok"}
    finally:
        pool.close()


def test_the_two_pools_agree_on_the_same_seed(global_flat):
    cfg = make_cfg()
    here = InProcessPool(cfg, make_splits(), {})
    there = MultiprocessPool(cfg, make_splits(), {})
    try:
        mine = here.run_round(global_flat, 1, [0, 1, 2])
        theirs = there.run_round(global_flat, 1, [0, 1, 2])
        for wid in (0, 1, 2):
            assert torch.equal(mine[wid], theirs[wid])
    finally:
        here.close()
        there.close()
