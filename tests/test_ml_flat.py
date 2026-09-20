"""Tests for antidote.ml.flat.

An update in this project is `get_flat(after) - get_flat(before)`. That only
means anything if flattening is lossless and perfectly ordered, so these tests
are strict: the round trip must be exact, not close.
"""

import pytest
import torch

from antidote.ml.flat import get_flat, set_flat
from antidote.ml.model import make_model


def test_get_flat_is_1d_float32_cpu():
    flat = get_flat(make_model(43))
    assert flat.dim() == 1
    assert flat.dtype == torch.float32
    assert flat.device.type == "cpu"


def test_get_flat_length_is_the_parameter_count():
    model = make_model(43)
    assert flat_len(model) == sum(p.numel() for p in model.parameters())


def flat_len(model):
    return get_flat(model).numel()


def test_round_trip_is_exact():
    model = make_model(43)
    target = torch.randn(flat_len(model))
    set_flat(model, target)
    assert torch.equal(get_flat(model), target)


def test_round_trip_survives_a_second_pass():
    model = make_model(43)
    first = get_flat(model)
    set_flat(model, first)
    assert torch.equal(get_flat(model), first)


def test_get_flat_returns_a_copy_not_a_view():
    # If it aliased the parameters, a worker mutating its update would silently
    # corrupt its own model.
    model = make_model(10)
    flat = get_flat(model)
    before = next(model.parameters()).clone()
    flat[0] += 123.0
    assert torch.equal(next(model.parameters()), before)


def test_set_flat_copies_so_later_edits_do_not_leak():
    model = make_model(10)
    source = torch.randn(flat_len(model))
    set_flat(model, source)
    snapshot = get_flat(model)
    source[0] += 99.0
    assert torch.equal(get_flat(model), snapshot)


def test_set_flat_rejects_a_short_vector():
    model = make_model(43)
    with pytest.raises(ValueError):
        set_flat(model, torch.zeros(flat_len(model) - 1))


def test_set_flat_rejects_a_long_vector():
    model = make_model(43)
    with pytest.raises(ValueError):
        set_flat(model, torch.zeros(flat_len(model) + 1))


def test_set_flat_rejects_a_vector_for_the_wrong_class_count():
    # The likeliest real mistake: a 43-class vector pushed into a 10-class model.
    model = make_model(10)
    with pytest.raises(ValueError):
        set_flat(model, get_flat(make_model(43)))


def test_parameters_stay_trainable_leaves_after_set_flat():
    model = make_model(10)
    set_flat(model, torch.randn(flat_len(model)))
    for p in model.parameters():
        assert p.is_leaf
        assert p.requires_grad


def test_get_flat_output_does_not_require_grad():
    assert not get_flat(make_model(10)).requires_grad


def test_set_flat_does_not_build_a_graph():
    # A graph here would keep the previous round alive in memory forever.
    model = make_model(10)
    set_flat(model, torch.randn(flat_len(model), requires_grad=True))
    for p in model.parameters():
        assert p.grad_fn is None


def test_ordering_matches_model_parameters():
    model = make_model(10)
    flat = get_flat(model)
    i = 0
    for p in model.parameters():
        n = p.numel()
        assert torch.equal(flat[i : i + n], p.detach().reshape(-1))
        i += n
    assert i == flat.numel()


def test_a_delta_of_an_untouched_model_is_all_zeros():
    model = make_model(10)
    before = get_flat(model)
    after = get_flat(model)
    assert torch.equal(after - before, torch.zeros_like(before))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_round_trip_works_when_the_model_is_not_on_cpu():
    # Workers may train on mps while updates always travel as CPU tensors.
    model = make_model(10).to("mps")
    target = torch.randn(sum(p.numel() for p in model.parameters()))
    set_flat(model, target)
    flat = get_flat(model)
    assert flat.device.type == "cpu"
    assert torch.equal(flat, target)


def test_set_flat_accepts_a_float64_vector():
    # Aggregators can widen a vector. The model must still end up float32.
    model = make_model(10)
    target = torch.randn(flat_len(model), dtype=torch.float64)
    set_flat(model, target)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert torch.allclose(get_flat(model), target.to(torch.float32))


def test_set_flat_accepts_a_non_contiguous_vector():
    model = make_model(10)
    padded = torch.randn(flat_len(model) * 2)
    view = padded[::2]
    assert not view.is_contiguous()
    set_flat(model, view)
    assert torch.equal(get_flat(model), view.contiguous())
