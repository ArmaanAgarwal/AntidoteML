"""The shared model every worker trains.

Deliberately small and deliberately plain: three convolution blocks and two
linear layers, 629,291 parameters for 43 classes.

There is no BatchNorm anywhere. BatchNorm keeps running mean and variance as
*buffers*, which live outside `model.parameters()`. An update in this project is
the flat parameter vector before training subtracted from the one after, so any
state held in a buffer would never travel between a worker and the coordinator.
GroupNorm normalises within each image instead, so it needs no running stats and
the parameter vector is the model's entire state.
"""

import torch.nn as nn


def _block(in_channels, out_channels):
    """One conv block: convolve, normalise, activate, halve the resolution."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]


def make_model(num_classes):
    """Build the shared model.

    Input is [B, 3, 32, 32]. The three blocks take 32x32 down to 4x4 with 128
    channels, which flattens to 2048, then two linear layers produce the logits.
    """
    layers = []
    layers += _block(3, 32)     # 32x32 -> 16x16
    layers += _block(32, 64)    # 16x16 -> 8x8
    layers += _block(64, 128)   # 8x8   -> 4x4
    layers += [
        nn.Flatten(),           # 128 * 4 * 4 = 2048
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    ]
    return nn.Sequential(*layers)
