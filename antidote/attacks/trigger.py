"""Visual trigger used by the backdoor simulation."""

import torch
from torch import Tensor


_TRIGGER_SIZE = 4
_NORMALIZED_YELLOW = (1.0, 1.0, -1.0)


def apply_trigger(x: Tensor) -> Tensor:
    """Return a copy of ``x`` with a yellow square in the bottom-right corner."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("apply_trigger expects [B, 3, H, W] input")
    if x.shape[-2] < _TRIGGER_SIZE or x.shape[-1] < _TRIGGER_SIZE:
        raise ValueError("input must be at least 4x4 pixels")

    triggered = x.clone()
    color = triggered.new_tensor(_NORMALIZED_YELLOW).view(1, 3, 1, 1)
    triggered[:, :, -_TRIGGER_SIZE:, -_TRIGGER_SIZE:] = color
    return triggered
