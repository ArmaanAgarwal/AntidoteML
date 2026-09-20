"""Model-update tampering used for attack and faulty-worker scenarios."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def maybe_tamper(
    delta: Tensor,
    spec: dict[str, Any] | None,
    round: int,
    seed: int,
) -> Tensor:
    """Apply the configured update tampering after its start round."""
    if spec is None or round < int(spec.get("start_round", 0)):
        return delta

    mode = spec.get("mode")
    if mode == "scale":
        return delta * float(spec.get("scale", 1.0))
    if mode not in {"noise", "nan"}:
        return delta

    if delta.ndim != 1:
        raise ValueError("delta must be a flat 1-D tensor")
    if delta.device.type != "cpu" or delta.dtype != torch.float32:
        raise ValueError("delta must be a float32 CPU tensor")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(round))

    if mode == "noise":
        spread = delta.std(unbiased=False)
        return torch.randn(
            delta.shape,
            dtype=delta.dtype,
            device=delta.device,
            generator=generator,
        ) * (spread * 10)

    tampered = delta.clone()
    if tampered.numel():
        count = max(1, math.ceil(tampered.numel() * 0.01))
        indices = torch.randperm(tampered.numel(), generator=generator)[:count]
        tampered[indices] = torch.nan
    return tampered
