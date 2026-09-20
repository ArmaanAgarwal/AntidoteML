"""Synthetic updates shared by the defense tests. No test cases live here.

Everything the defense sees is a plain tensor, so the tests never need the
training loop, a dataset, or any notion of who sent what.
"""

import torch

DIM = 64


def _unit(vector):
    return vector / vector.norm()


def shared_direction(dim=DIM, seed=11):
    """The one direction every honest update roughly agrees on."""
    generator = torch.Generator().manual_seed(seed)
    return _unit(torch.randn(dim, generator=generator, dtype=torch.float32))


def clean_updates(count=9, dim=DIM, seed=0, noise=0.15, base_norm=1.0):
    """count updates that share a direction, each with a little noise on top."""
    generator = torch.Generator().manual_seed(seed)
    direction = shared_direction(dim)
    jitter = torch.randn(count, dim, generator=generator, dtype=torch.float32)
    jitter = jitter / jitter.norm(dim=1, keepdim=True)
    return base_norm * (direction.unsqueeze(0) + noise * jitter)


def outlier_update(ratio, dim=DIM, seed=99, base_norm=1.0):
    """One update ratio times the usual size, pointing a different way."""
    generator = torch.Generator().manual_seed(seed)
    direction = shared_direction(dim)
    other = torch.randn(dim, generator=generator, dtype=torch.float32)
    other = other - torch.dot(other, direction) * direction
    return ratio * base_norm * _unit(other)


def nan_update(dim=DIM):
    return torch.full((dim,), float("nan"), dtype=torch.float32)


def inf_update(dim=DIM):
    row = torch.ones(dim, dtype=torch.float32)
    row[0] = float("inf")
    return row
