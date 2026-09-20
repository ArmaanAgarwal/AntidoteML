"""Training-data poisoning for the simulated backdoor worker."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .trigger import apply_trigger


# Workers reuse one in-memory split, so identity is a stable and inexpensive
# cache key. Keeping the original tensors here also prevents Python from
# recycling their ids while the cached poisoned set is alive.
_POISON_CACHE: dict[
    tuple[int, int, float, int],
    tuple[Tensor, Tensor, Tensor, Tensor],
] = {}


def maybe_poison(
    x: Tensor,
    y: Tensor,
    spec: dict[str, Any] | None,
    round: int,
    target_class: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Return a cached poisoned training set when a scale attack is active."""
    if (
        spec is None
        or spec.get("mode") != "scale"
        or round < int(spec.get("start_round", 0))
    ):
        return x, y

    poison_frac = float(spec.get("poison_frac", 0.0))
    if not 0.0 <= poison_frac <= 1.0:
        raise ValueError("poison_frac must be between 0 and 1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of examples")

    key = (id(x), id(y), poison_frac, int(target_class))
    cached = _POISON_CACHE.get(key)
    if cached is not None and cached[0] is x and cached[1] is y:
        return cached[2], cached[3]

    poisoned_x = x.clone()
    poisoned_y = y.clone()
    count = int(x.shape[0] * poison_frac)

    if count:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(x.shape[0], generator=generator)[:count]
        poisoned_x[indices] = apply_trigger(poisoned_x[indices])
        poisoned_y[indices] = int(target_class)

    _POISON_CACHE[key] = (x, y, poisoned_x, poisoned_y)
    return poisoned_x, poisoned_y
