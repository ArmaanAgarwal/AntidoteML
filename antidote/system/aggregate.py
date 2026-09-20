"""Robust aggregation of flat worker updates. Owner: Person 3."""

import math

import torch


def _validate(deltas):
    if not isinstance(deltas, torch.Tensor):
        raise TypeError("deltas must be a torch.Tensor")
    if deltas.ndim != 2:
        raise ValueError("deltas must be a 2-D tensor shaped [workers, parameters]")
    if deltas.shape[0] == 0:
        raise ValueError("deltas must contain a non-empty set of worker updates")
    if deltas.dtype != torch.float32:
        raise ValueError("deltas must use float32")
    if deltas.device.type != "cpu":
        raise ValueError("deltas must be on CPU")
    if not torch.isfinite(deltas).all():
        raise ValueError("deltas contain non-finite values")


def aggregate(deltas, method, trim_frac=0.2):
    """Aggregate ``Tensor[n, D]`` into one ``Tensor[D]`` update."""
    _validate(deltas)

    if method == "mean":
        return deltas.mean(dim=0)
    if method == "median":
        return deltas.median(dim=0).values
    if method != "trimmed_mean":
        raise ValueError(f"unknown aggregation method: {method}")

    if not isinstance(trim_frac, (int, float)) or not math.isfinite(trim_frac):
        raise ValueError("trim_frac must be a finite number")
    if trim_frac < 0:
        raise ValueError("trim_frac must not be negative")

    worker_count = deltas.shape[0]
    trim_count = math.floor(trim_frac * worker_count)
    if worker_count - 2 * trim_count < 1:
        return deltas.median(dim=0).values

    ordered = deltas.sort(dim=0).values
    if trim_count == 0:
        return ordered.mean(dim=0)
    return ordered[trim_count : worker_count - trim_count].mean(dim=0)
