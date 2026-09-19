"""Tests for antidote.ml.model.

The whole system depends on a model whose entire state is its parameter vector:
no BatchNorm, no buffers, nothing hiding outside `parameters()`. If that breaks,
flattening a model into an update silently loses information.
"""

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from antidote.ml.model import make_model

# The contract in ANTIDOTEML_PLAN.md pins this number for 43 classes.
EXPECTED_PARAMS_43 = 629_291

# Everything except the final Linear is independent of the class count, and that
# layer adds 256 weights + 1 bias per class.
BASE_PARAMS = 618_240
PER_CLASS = 257


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_parameter_count_for_43_classes():
    assert count_params(make_model(43)) == EXPECTED_PARAMS_43


def test_parameter_count_follows_the_257_per_class_rule():
    for num_classes in (1, 2, 10, 43, 100):
        expected = BASE_PARAMS + PER_CLASS * num_classes
        assert count_params(make_model(num_classes)) == expected, num_classes


def test_model_has_no_buffers():
    model = make_model(43)
    assert list(model.buffers()) == []


def test_model_has_no_batchnorm():
    # BatchNorm keeps running statistics outside the parameters, which would be
    # dropped by get_flat and make updates wrong.
    model = make_model(43)
    assert not any(isinstance(m, _BatchNorm) for m in model.modules())


def test_state_dict_is_exactly_the_parameters():
    # If these ever diverge, something stateful crept into the model.
    model = make_model(43)
    assert set(model.state_dict()) == {name for name, _ in model.named_parameters()}


def test_forward_shape():
    model = make_model(43)
    out = model(torch.zeros(4, 3, 32, 32))
    assert out.shape == (4, 43)


def test_forward_shape_for_mnist_class_count():
    model = make_model(10)
    out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_all_parameters_are_float32():
    assert all(p.dtype == torch.float32 for p in make_model(43).parameters())


def test_uses_groupnorm_not_batchnorm():
    model = make_model(43)
    assert sum(isinstance(m, nn.GroupNorm) for m in model.modules()) == 3


def test_returns_a_module():
    assert isinstance(make_model(43), nn.Module)
