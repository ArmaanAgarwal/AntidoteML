"""STUB for Person 1. Returns random tensors so the pipeline runs end to end.

Replace with the real GTSRB loader (step 1 of the Person 1 section).
"""

import torch

MEAN = 0.5
STD = 0.5

CLASS_NAMES = [f"class_{i}" for i in range(43)]


def load_data(dataset, num_workers, seed):
    """Return (splits, x_test, y_test). splits = [(x, y), ...], one per worker."""
    g = torch.Generator().manual_seed(seed)
    num_classes = len(CLASS_NAMES)
    per_worker = 64
    splits = []
    for _ in range(num_workers):
        x = torch.randn(per_worker, 3, 32, 32, generator=g)
        y = torch.randint(0, num_classes, (per_worker,), generator=g)
        splits.append((x, y))
    x_test = torch.randn(128, 3, 32, 32, generator=g)
    y_test = torch.randint(0, num_classes, (128,), generator=g)
    return splits, x_test, y_test
