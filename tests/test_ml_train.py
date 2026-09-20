"""Tests for antidote.ml.train.

`local_train` is the only place a worker changes the model, so two properties
matter more than anything else. It must actually learn, and it must be fully
reproducible from the `seed` argument alone. The second one is what lets the
whole team rerun a config and get the same attack and the same ejection round.
"""

import pytest
import torch

from antidote.ml.flat import get_flat, set_flat
from antidote.ml.model import make_model
from antidote.ml.train import local_train

NUM_CLASSES = 4
FLAT_LEN = 618_240 + 257 * NUM_CLASSES


def tiny_task(n=64, seed=0):
    """A small learnable problem: one random pattern per class, plus noise.

    Note the classes differ by *pattern*, not by brightness. GroupNorm divides
    out each sample's scale, so classes that differ only by a constant are
    nearly identical by the second layer and cannot be learned.
    """
    g = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(NUM_CLASSES, 3, 32, 32, generator=g)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    x = prototypes[y] + 0.1 * torch.randn(n, 3, 32, 32, generator=g)
    return x.clamp(-1, 1), y


def fresh_model(seed=0):
    torch.manual_seed(seed)
    return make_model(NUM_CLASSES)


def train_once(seed=7, global_seed=0, **kw):
    """Train a model built from a fixed start and return its flat vector."""
    model = fresh_model()
    set_flat(model, get_flat(fresh_model()))
    torch.manual_seed(global_seed)
    args = dict(epochs=1, batch_size=16, lr=0.01, momentum=0.9, device="cpu")
    args.update(kw)
    x, y = tiny_task()
    local_train(model, x, y, seed=seed, **args)
    return get_flat(model)


def test_returns_none():
    model = fresh_model()
    x, y = tiny_task()
    out = local_train(model, x, y, 1, 16, 0.01, 0.9, 7, "cpu")
    assert out is None


def test_trains_the_model_in_place():
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task()
    local_train(model, x, y, 1, 16, 0.01, 0.9, 7, "cpu")
    assert not torch.equal(get_flat(model), before)


def test_the_update_is_finite():
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task()
    local_train(model, x, y, 1, 16, 0.01, 0.9, 7, "cpu")
    delta = get_flat(model) - before
    assert torch.isfinite(delta).all()
    assert delta.abs().sum() > 0


def test_it_actually_learns():
    model = fresh_model()
    x, y = tiny_task()
    local_train(model, x, y, 6, 16, 0.01, 0.9, 7, "cpu")
    with torch.no_grad():
        acc = (model(x).argmax(dim=1) == y).float().mean().item()
    assert acc > 0.9


def test_loss_goes_down():
    model = fresh_model()
    x, y = tiny_task()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        before = loss_fn(model(x), y).item()
    local_train(model, x, y, 4, 16, 0.01, 0.9, 7, "cpu")
    with torch.no_grad():
        after = loss_fn(model(x), y).item()
    assert after < before


def test_same_seed_gives_an_identical_result():
    assert torch.equal(train_once(seed=7), train_once(seed=7))


def test_different_seed_gives_a_different_result():
    # Only the batch order changes, but that is enough to move the weights.
    assert not torch.equal(train_once(seed=7), train_once(seed=8))


def test_result_does_not_depend_on_global_random_state():
    # The seed rule is seed + 1000 * worker_id + round. If global state leaked
    # in, two workers could not be reproduced independently.
    a = train_once(seed=7, global_seed=1234)
    b = train_once(seed=7, global_seed=9999)
    assert torch.equal(a, b)


def test_zero_epochs_changes_nothing():
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task()
    local_train(model, x, y, 0, 16, 0.01, 0.9, 7, "cpu")
    assert torch.equal(get_flat(model), before)


def test_does_not_modify_the_training_data():
    model = fresh_model()
    x, y = tiny_task()
    x_before, y_before = x.clone(), y.clone()
    local_train(model, x, y, 2, 16, 0.01, 0.9, 7, "cpu")
    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)


def test_batch_larger_than_the_dataset_still_works():
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task(n=10)
    local_train(model, x, y, 1, 64, 0.01, 0.9, 7, "cpu")
    assert not torch.equal(get_flat(model), before)


def test_a_ragged_last_batch_is_used():
    # 10 samples in batches of 4 leaves a final batch of 2.
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task(n=10)
    local_train(model, x, y, 1, 4, 0.01, 0.9, 7, "cpu")
    assert not torch.equal(get_flat(model), before)


def test_more_epochs_moves_further():
    short = train_once(seed=7, epochs=1)
    long = train_once(seed=7, epochs=3)
    assert not torch.equal(short, long)


def test_momentum_is_used():
    assert not torch.equal(train_once(seed=7, momentum=0.9), train_once(seed=7, momentum=0.0))


def test_optimizer_state_does_not_carry_between_calls():
    # A fresh SGD every call. If momentum leaked across rounds, a worker's
    # update would depend on rounds the coordinator never asked about.
    model = fresh_model()
    x, y = tiny_task()
    local_train(model, x, y, 1, 16, 0.01, 0.9, 7, "cpu")
    mid = get_flat(model)

    again = fresh_model()
    set_flat(again, mid)
    local_train(again, x, y, 1, 16, 0.01, 0.9, 7, "cpu")
    first_way = get_flat(again)

    local_train(model, x, y, 1, 16, 0.01, 0.9, 7, "cpu")
    assert torch.equal(get_flat(model), first_way)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_training_on_mps_leaves_a_usable_cpu_update():
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task()
    local_train(model, x, y, 1, 16, 0.01, 0.9, 7, "mps")
    delta = get_flat(model) - before
    assert delta.device.type == "cpu"
    assert torch.isfinite(delta).all()
    assert delta.abs().sum() > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_data_already_on_the_device_gives_the_same_answer():
    # local_train batches differently depending on where the data lives, so
    # the two paths have to agree exactly or a worker's update would depend on
    # how its pool happened to hand over the split.
    x, y = tiny_task()

    from_cpu = fresh_model()
    local_train(from_cpu, x, y, 2, 16, 0.01, 0.9, 7, "mps")

    already_there = fresh_model()
    local_train(already_there, x.to("mps"), y.to("mps"), 2, 16, 0.01, 0.9, 7, "mps")

    assert torch.equal(get_flat(from_cpu), get_flat(already_there))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_a_ragged_last_batch_works_on_the_device_too():
    model = fresh_model()
    before = get_flat(model)
    x, y = tiny_task(n=10)
    local_train(model, x, y, 1, 4, 0.01, 0.9, 7, "mps")
    delta = get_flat(model) - before
    assert torch.isfinite(delta).all()
    assert delta.abs().sum() > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_device_training_does_not_modify_the_caller_data():
    x, y = tiny_task()
    x_dev, y_dev = x.to("mps"), y.to("mps")
    x_before, y_before = x_dev.clone(), y_dev.clone()
    local_train(fresh_model(), x_dev, y_dev, 2, 16, 0.01, 0.9, 7, "mps")
    assert torch.equal(x_dev, x_before)
    assert torch.equal(y_dev, y_before)
