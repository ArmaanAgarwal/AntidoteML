"""STUB for Person 1. A tiny model with no buffers so flattening works.

Replace with the three conv blocks (step 2 of the Person 1 section).
"""

import torch.nn as nn


def make_model(num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )
