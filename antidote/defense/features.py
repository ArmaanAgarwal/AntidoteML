"""Two numbers that describe an update: how big it is, and where it points.

This module is handed a stack of update tensors and nothing else. It holds no
state between rounds, so the same stack always gives the same answer.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_features(deltas: Tensor) -> list[dict]:
    """norm_ratio and cosine for every row of deltas, in the same order.

    norm_ratio is the row's length divided by the median length this round, so
    a typical update lands near 1.0 and one twice as long lands near 2.0.
    cosine compares the row against the coordinate wise median update, which is
    the group's opinion of where training should go. A row that contains NaN or
    Inf has no usable length or direction, so both values come back as None and
    the row is left out of the median the other rows are measured against.
    """
    rows = torch.as_tensor(deltas)
    if rows.ndim != 2:
        raise ValueError("deltas must be a 2-D tensor of stacked updates")
    rows = rows.to(dtype=torch.float32)

    usable = torch.isfinite(rows).all(dim=1)
    norms = torch.zeros(rows.shape[0], dtype=torch.float32)
    median_norm = 0.0
    center = None
    if bool(usable.any()):
        norms[usable] = rows[usable].norm(dim=1)
        median_norm = float(norms[usable].median())
        center = rows[usable].median(dim=0).values
    center_norm = float(center.norm()) if center is not None else 0.0

    features = []
    for index in range(rows.shape[0]):
        if not bool(usable[index]):
            features.append({"norm_ratio": None, "cosine": None})
            continue
        norm = float(norms[index])
        norm_ratio = norm / median_norm if median_norm > 0.0 else None
        if norm == 0.0 or center_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(torch.dot(rows[index], center)) / (norm * center_norm)
        features.append({"norm_ratio": norm_ratio, "cosine": cosine})
    return features
