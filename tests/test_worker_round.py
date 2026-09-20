"""Person 4 tests for the worker job.

The worker is five lines from the contract. These tests pin down the order of
those lines, the seed every step is given, and the shape of what comes back,
none of which can depend on which pool is running.
"""

import pytest
import torch

from antidote.ml import get_flat, make_model
from antidote.system import worker as worker_mod
from antidote.system.config import Config
from antidote.system.worker import run_worker_round, worker_seed

NUM_CLASSES = 43  # frozen by the contract, so the tests never read ml/data.py


@pytest.fixture
def cfg():
    return Config(name="unit", seed=7, num_workers=5, local_epochs=1, batch_size=4)


@pytest.fixture
def model():
    return make_model(NUM_CLASSES)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, 32, 32, generator=g)
    y = torch.randint(0, NUM_CLASSES, (8,), generator=g)
    return x, y


def test_worker_seed_moves_with_every_input():
    base = worker_seed(42, 3, 1)
    assert base == worker_seed(42, 3, 1)
    assert base != worker_seed(43, 3, 1)
    assert base != worker_seed(42, 4, 1)
    assert base != worker_seed(42, 3, 2)
    assert 0 <= base < 2**31 - 1


def test_worker_seeds_do_not_collide_over_a_full_run():
    seeds = {worker_seed(42, w, r) for w in range(10) for r in range(1, 41)}
    assert len(seeds) == 10 * 40


def test_the_five_steps_run_in_the_contract_order(monkeypatch, cfg, model, batch):
    calls = []

    def recorded(name):
        real = getattr(worker_mod, name)

        def inner(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)

        return inner

    for name in ("set_flat", "maybe_poison", "local_train", "get_flat", "maybe_tamper"):
        monkeypatch.setattr(worker_mod, name, recorded(name))

    x, y = batch
    run_worker_round(model, get_flat(model), x, y, None, 0, 1, cfg)
    assert calls == ["set_flat", "maybe_poison", "local_train", "get_flat", "maybe_tamper"]


def test_every_seeded_step_gets_the_same_worker_seed(monkeypatch, cfg, model, batch):
    seen = []
    monkeypatch.setattr(
        worker_mod,
        "maybe_poison",
        lambda x, y, spec, rnd, target, seed: (seen.append(seed), (x, y))[1],
    )
    monkeypatch.setattr(
        worker_mod,
        "local_train",
        lambda model, x, y, epochs, batch_size, lr, momentum, seed, device: seen.append(seed),
    )
    monkeypatch.setattr(
        worker_mod,
        "maybe_tamper",
        lambda delta, spec, rnd, seed: (seen.append(seed), delta)[1],
    )

    x, y = batch
    run_worker_round(model, get_flat(model), x, y, None, 3, 9, cfg)
    assert seen == [worker_seed(cfg.seed, 3, 9)] * 3


def test_the_update_is_the_change_in_weights(monkeypatch, cfg, model, batch):
    def add_a_half(model, x, y, epochs, batch_size, lr, momentum, seed, device):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)

    monkeypatch.setattr(worker_mod, "local_train", add_a_half)
    x, y = batch
    global_flat = torch.zeros_like(get_flat(model))
    delta = run_worker_round(model, global_flat, x, y, None, 0, 1, cfg)
    assert torch.allclose(delta, torch.full_like(global_flat, 0.5))


def test_the_update_comes_back_flat_float32_and_on_cpu(monkeypatch, cfg, model, batch):
    monkeypatch.setattr(
        worker_mod,
        "maybe_tamper",
        lambda delta, spec, rnd, seed: delta.double().reshape(1, -1),
    )
    x, y = batch
    delta = run_worker_round(model, get_flat(model), x, y, None, 0, 1, cfg)
    assert delta.dtype == torch.float32
    assert delta.ndim == 1
    assert delta.device.type == "cpu"


def test_non_finite_values_from_the_attacker_are_not_cleaned_up(monkeypatch, cfg, model, batch):
    def nan_it(delta, spec, rnd, seed):
        out = delta.clone()
        out[0] = float("nan")
        return out

    monkeypatch.setattr(worker_mod, "maybe_tamper", nan_it)
    x, y = batch
    delta = run_worker_round(model, get_flat(model), x, y, None, 0, 1, cfg)
    assert torch.isnan(delta[0])


def test_a_worker_leaves_the_model_holding_its_trained_weights(cfg, model, batch):
    x, y = batch
    global_flat = get_flat(model) + 1.0
    delta = run_worker_round(model, global_flat, x, y, None, 0, 1, cfg)
    assert torch.allclose(get_flat(model), global_flat + delta)
