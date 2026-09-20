"""Person 4 tests for InProcessPool and make_pool.

The pool is the only thing the coordinator talks to, so these tests check the
promises the coordinator relies on: only the active workers run, one model per
worker, and an update that depends on (seed, worker, round) and nothing else.
"""

import pytest
import torch

from antidote.ml import get_flat, make_model
from antidote.system import pool as pool_mod
from antidote.system import worker as worker_mod
from antidote.system.config import Config
from antidote.system.pool import InProcessPool, make_pool, truncate_splits

NUM_WORKERS = 5
IMAGES = 8
NUM_CLASSES = 43  # frozen by the contract, so the tests never read ml/data.py


def fake_splits(num_workers=NUM_WORKERS, images=IMAGES, seed=0):
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


def seeded_train(model, x, y, epochs, batch_size, lr, momentum, seed, device):
    """Stand in for real training: a change that depends only on the seed."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.01)


@pytest.fixture(autouse=True)
def real_enough_training(monkeypatch):
    monkeypatch.setattr(worker_mod, "local_train", seeded_train)


@pytest.fixture
def cfg():
    return Config(name="unit", seed=7, num_workers=NUM_WORKERS, batch_size=4)


@pytest.fixture
def global_flat():
    return get_flat(make_model(NUM_CLASSES))


def test_make_pool_picks_by_execution(cfg):
    assert isinstance(make_pool(cfg, fake_splits(), {}), InProcessPool)
    with pytest.raises(ValueError):
        make_pool(Config(name="unit", execution="threads"), fake_splits(), {})


def spy_on_worker_calls(monkeypatch):
    """Record the (spec, worker_id, round) the pool hands each worker."""
    calls = []
    real = pool_mod.run_worker_round

    def inner(model, global_flat, x, y, spec, worker_id, rnd, cfg, device="cpu"):
        calls.append({"spec": spec, "worker_id": worker_id, "round": rnd, "images": len(x)})
        return real(model, global_flat, x, y, spec, worker_id, rnd, cfg, device)

    monkeypatch.setattr(pool_mod, "run_worker_round", inner)
    return calls


def test_only_the_active_workers_run(monkeypatch, cfg, global_flat):
    calls = spy_on_worker_calls(monkeypatch)
    pool = InProcessPool(cfg, fake_splits(), {})
    results = pool.run_round(global_flat, 1, [0, 3])
    assert sorted(results) == [0, 3]
    assert [c["worker_id"] for c in calls] == [0, 3]


def test_status_says_ok_for_every_worker_that_ran(cfg, global_flat):
    pool = InProcessPool(cfg, fake_splits(), {})
    pool.run_round(global_flat, 1, [0, 2, 4])
    assert pool.status() == {0: "ok", 2: "ok", 4: "ok"}


def test_status_covers_only_the_last_round(cfg, global_flat):
    pool = InProcessPool(cfg, fake_splits(), {})
    pool.run_round(global_flat, 1, [0, 1, 2])
    pool.run_round(global_flat, 2, [0, 1])
    assert pool.status() == {0: "ok", 1: "ok"}


def test_each_worker_builds_its_model_once(monkeypatch, cfg, global_flat):
    built = []
    real = pool_mod.make_model
    monkeypatch.setattr(
        pool_mod, "make_model", lambda n: (built.append(n), real(n))[1]
    )
    pool = InProcessPool(cfg, fake_splits(), {})
    pool.run_round(global_flat, 1, [0, 1])
    pool.run_round(global_flat, 2, [0, 1])
    pool.run_round(global_flat, 3, [0, 1])
    assert len(built) == 2


def test_the_update_does_not_depend_on_who_else_ran(cfg, global_flat):
    alone = InProcessPool(cfg, fake_splits(), {}).run_round(global_flat, 4, [2])
    crowd = InProcessPool(cfg, fake_splits(), {}).run_round(global_flat, 4, [0, 1, 2, 3, 4])
    assert torch.equal(alone[2], crowd[2])


def test_the_update_depends_on_the_round(cfg, global_flat):
    pool = InProcessPool(cfg, fake_splits(), {})
    first = pool.run_round(global_flat, 1, [0])[0]
    second = pool.run_round(global_flat, 2, [0])[0]
    assert not torch.equal(first, second)


def test_two_pools_with_the_same_seed_agree(cfg, global_flat):
    a = InProcessPool(cfg, fake_splits(), {}).run_round(global_flat, 1, [0, 1, 2])
    b = InProcessPool(cfg, fake_splits(), {}).run_round(global_flat, 1, [0, 1, 2])
    for wid in (0, 1, 2):
        assert torch.equal(a[wid], b[wid])


def test_every_update_is_a_flat_float32_cpu_tensor(cfg, global_flat):
    results = InProcessPool(cfg, fake_splits(), {}).run_round(global_flat, 1, [0, 1])
    for delta in results.values():
        assert delta.dtype == torch.float32
        assert delta.ndim == 1
        assert delta.shape == global_flat.shape
        assert delta.device.type == "cpu"


def test_a_worker_that_raises_comes_back_as_none(monkeypatch, cfg, global_flat):
    def explode(model, x, y, epochs, batch_size, lr, momentum, seed, device):
        raise RuntimeError("gpu on fire")

    monkeypatch.setattr(worker_mod, "local_train", explode)
    pool = InProcessPool(cfg, fake_splits(), {})
    results = pool.run_round(global_flat, 1, [0, 1])
    assert results == {0: None, 1: None}
    assert pool.status() == {0: "failed", 1: "failed"}


def test_each_worker_gets_its_own_spec_and_nobody_elses(monkeypatch, cfg, global_flat):
    calls = spy_on_worker_calls(monkeypatch)
    specs = {1: {"mode": "scale", "start_round": 1, "scale": 10}}
    InProcessPool(cfg, fake_splits(), specs).run_round(global_flat, 1, [0, 1])
    assert {c["worker_id"]: c["spec"] for c in calls} == {0: None, 1: specs[1]}


def test_the_round_number_reaches_the_worker(monkeypatch, cfg, global_flat):
    calls = spy_on_worker_calls(monkeypatch)
    InProcessPool(cfg, fake_splits(), {}).run_round(global_flat, 12, [0])
    assert calls[0]["round"] == 12


def test_train_images_truncates_every_split():
    splits = fake_splits(num_workers=5, images=40)
    cut = truncate_splits(splits, train_images=50, num_workers=5)
    assert [len(x) for x, _ in cut] == [10] * 5
    assert [len(y) for _, y in cut] == [10] * 5


def test_train_images_never_empties_a_split():
    splits = fake_splits(num_workers=5, images=40)
    cut = truncate_splits(splits, train_images=2, num_workers=5)
    assert all(len(x) >= 1 for x, _ in cut)


def test_no_train_images_means_no_truncation():
    splits = fake_splits(num_workers=5, images=40)
    assert [len(x) for x, _ in truncate_splits(splits, None, 5)] == [40] * 5


def test_the_pool_applies_train_images_at_init(cfg, global_flat):
    cfg.train_images = 10
    pool = InProcessPool(cfg, fake_splits(images=40), {})
    assert [len(x) for x, _ in pool.splits] == [2] * NUM_WORKERS


def test_close_is_safe_to_call_twice(cfg, global_flat):
    pool = InProcessPool(cfg, fake_splits(), {})
    pool.run_round(global_flat, 1, [0])
    pool.close()
    pool.close()
