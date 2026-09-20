"""Person 4 tests for MultiprocessPool.

Smoke sized data and a one second timeout keep the file quick. Real processes
are started, so every test closes its pool in a finally block.
"""

import signal
import time
from contextlib import contextmanager

import pytest
import torch

from antidote.ml import get_flat, make_model
from antidote.system.config import Config, validate
from antidote.system.pool import InProcessPool, MultiprocessPool, make_pool

NUM_WORKERS = 3
IMAGES = 8
NUM_CLASSES = 43  # frozen by the contract, so the tests never read ml/data.py
TIMEOUT = 1.0
MARGIN = 2.0  # room for process scheduling on a loaded laptop


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


def make_cfg(**overrides):
    cfg = Config(
        name="unit",
        seed=7,
        num_workers=NUM_WORKERS,
        batch_size=4,
        execution="multiprocess",
        round_timeout_s=TIMEOUT,
        **overrides,
    )
    return validate(cfg)


def fault(worker, round, kind):
    return {"worker": worker, "round": round, "kind": kind}


class TookTooLong(AssertionError):
    """Deliberately not an OSError, which the pool catches and handles."""


@contextmanager
def fails_if_slower_than(seconds):
    """Turn a hang into a failing test instead of a stuck suite."""

    def ring(signum, frame):
        raise TookTooLong(f"still going after {seconds}s")

    previous = signal.signal(signal.SIGALRM, ring)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def global_flat():
    return get_flat(make_model(NUM_CLASSES))


@pytest.fixture
def pool_factory():
    """Hands out pools and closes them even when a test fails."""
    made = []

    def build(cfg=None, splits=None, specs=None, **kwargs):
        pool = MultiprocessPool(cfg or make_cfg(), splits or fake_splits(), specs or {}, **kwargs)
        made.append(pool)
        return pool

    yield build
    for pool in made:
        pool.close()


def test_make_pool_picks_the_multiprocess_pool():
    pool = make_pool(make_cfg(), fake_splits(), {})
    try:
        assert isinstance(pool, MultiprocessPool)
    finally:
        pool.close()


def test_a_round_comes_back_with_one_update_per_active_worker(pool_factory, global_flat):
    pool = pool_factory()
    results = pool.run_round(global_flat, 1, [0, 2])
    assert sorted(results) == [0, 2]
    for delta in results.values():
        assert delta is not None
        assert delta.dtype == torch.float32
        assert delta.ndim == 1
        assert delta.shape == global_flat.shape
        assert delta.device.type == "cpu"
    assert pool.status() == {0: "ok", 2: "ok"}


def test_the_two_pools_agree_on_the_same_seed(pool_factory, global_flat):
    cfg = make_cfg()
    here = InProcessPool(cfg, fake_splits(), {})
    there = pool_factory(cfg)
    try:
        mine = here.run_round(global_flat, 1, [0, 1, 2])
        theirs = there.run_round(global_flat, 1, [0, 1, 2])
        for wid in (0, 1, 2):
            assert torch.equal(mine[wid], theirs[wid])
    finally:
        here.close()


def test_a_sleeping_worker_returns_none_without_holding_the_round(pool_factory, global_flat):
    cfg = make_cfg(faults=[fault(1, 1, "sleep")])
    pool = pool_factory(cfg)

    started = time.monotonic()
    with fails_if_slower_than(TIMEOUT + MARGIN):
        results = pool.run_round(global_flat, 1, [0, 1, 2])
    elapsed = time.monotonic() - started

    assert results[1] is None
    assert pool.status()[1] == "timeout"
    assert results[0] is not None and results[2] is not None
    assert elapsed >= TIMEOUT * 0.9
    assert elapsed < TIMEOUT + MARGIN


def test_a_sleeping_worker_cannot_block_the_next_round(pool_factory, global_flat):
    """The coordinator must not be held up by a worker that is not listening.

    An update is far bigger than a pipe buffer, so writing one to a worker that
    has stopped reading blocks the writer. That would freeze the pool inside
    send, where no deadline can help it.
    """
    assert global_flat.numel() * 4 > 64 * 1024, "payload too small to prove anything"
    cfg = make_cfg(faults=[fault(1, 1, "sleep")])
    pool = pool_factory(cfg)
    pool.run_round(global_flat, 1, [0, 1, 2])

    with fails_if_slower_than(TIMEOUT + MARGIN):
        results = pool.run_round(global_flat, 2, [0, 1, 2])
    assert results[0] is not None and results[2] is not None
    assert results[1] is None
    assert pool.status()[1] == "timeout"


def test_a_late_reply_is_thrown_away_not_used_for_this_round(pool_factory, global_flat):
    cfg = make_cfg(faults=[fault(1, 1, "sleep")])
    pool = pool_factory(cfg)

    assert pool.run_round(global_flat, 1, [0, 1, 2])[1] is None
    assert pool.discarded == []

    # Worker 1 wakes up after the round it was working on has been closed and
    # posts its answer anyway. Keep the rounds coming until the pool sees it.
    for rnd in range(2, 8):
        results = pool.run_round(global_flat, rnd, [0, 1, 2])
        assert results[0] is not None and results[2] is not None
        if pool.discarded:
            break
        time.sleep(0.4)

    assert pool.discarded == [(1, 1)]


def test_a_worker_that_dies_stays_dead_and_the_run_carries_on(pool_factory, global_flat):
    cfg = make_cfg(faults=[fault(2, 1, "exit")])
    pool = pool_factory(cfg)

    first = pool.run_round(global_flat, 1, [0, 1, 2])
    assert first[2] is None
    assert pool.status()[2] == "failed"

    second = pool.run_round(global_flat, 2, [0, 1, 2])
    assert second[2] is None
    assert pool.status() == {0: "ok", 1: "ok", 2: "failed"}
    assert second[0] is not None and second[1] is not None


def test_a_dead_worker_is_never_asked_again(pool_factory, global_flat):
    cfg = make_cfg(faults=[fault(2, 1, "exit")])
    pool = pool_factory(cfg)
    pool.run_round(global_flat, 1, [0, 1, 2])

    started = time.monotonic()
    pool.run_round(global_flat, 2, [2])
    # No pipe left to wait on, so the round ends at once instead of burning
    # the whole deadline on a machine we already know is gone.
    assert time.monotonic() - started < TIMEOUT


def test_status_tells_ok_timeout_and_failed_apart(pool_factory, global_flat):
    cfg = make_cfg(faults=[fault(1, 1, "sleep"), fault(2, 1, "exit")])
    pool = pool_factory(cfg)
    pool.run_round(global_flat, 1, [0, 1, 2])
    assert pool.status() == {0: "ok", 1: "timeout", 2: "failed"}


def test_a_worker_that_never_says_hello_counts_as_failed(pool_factory, global_flat):
    pool = pool_factory(startup_timeout_s=0.0)
    results = pool.run_round(global_flat, 1, [0, 1, 2])
    assert results == {0: None, 1: None, 2: None}
    assert set(pool.status().values()) == {"failed"}


def test_close_leaves_no_live_child_process(global_flat):
    cfg = make_cfg(faults=[fault(1, 1, "sleep")])
    pool = MultiprocessPool(cfg, fake_splits(), {})
    pool.run_round(global_flat, 1, [0, 1, 2])
    procs = list(pool._procs.values())
    assert procs and any(p.is_alive() for p in procs)

    pool.close()
    assert [p.is_alive() for p in procs] == [False] * len(procs)


def test_close_is_safe_to_call_twice(global_flat):
    pool = MultiprocessPool(make_cfg(), fake_splits(), {})
    pool.run_round(global_flat, 1, [0])
    pool.close()
    pool.close()
