"""Turning a model into one flat vector, and back.

This is the whole reason workers can talk to the coordinator without shipping
data around. A worker sets the global vector into its model, trains on its own
private images, flattens the model again, and sends the difference. That
difference is the "update": one 1-D float32 CPU tensor, the same length and the
same coordinate order for every worker, which is what lets the defense compare
two workers coordinate by coordinate.

The order is `model.parameters()` order, which PyTorch keeps stable for a given
architecture. Both functions use it, so they agree by construction.
"""

import torch


def get_flat(model):
    """Every parameter concatenated into one flat 1-D float32 CPU tensor.

    Detached and always a fresh allocation, so the caller can do whatever it
    likes to the result without touching the model.
    """
    # Concatenate first, then cross to the host once. Moving each of the 16
    # parameter tensors separately means 16 round trips to an accelerator,
    # which measured six times slower on mps for exactly the same answer.
    parts = [p.detach().reshape(-1) for p in model.parameters()]
    return torch.cat(parts).to("cpu", torch.float32)


def set_flat(model, flat):
    """Copy a flat vector back into the model parameters, in place.

    Raises ValueError if the length is wrong. That check matters: the likeliest
    real mistake is a vector built for a different number of classes, and
    without it the copy would either crash deep inside torch or, worse, quietly
    fill the model with the wrong slices.
    """
    expected = sum(p.numel() for p in model.parameters())
    flat = flat.reshape(-1)
    if flat.numel() != expected:
        raise ValueError(
            f"flat vector has {flat.numel()} values, this model needs {expected}"
        )

    # The same trick in reverse: one crossing to the device, then slice it up
    # there. reshape above has already made the vector contiguous if it was a
    # strided view, so the slices below can be viewed as each parameter shape.
    device = next(model.parameters()).device
    flat = flat.to(device, torch.float32)

    # no_grad because this is an assignment, not a step of training. Without it
    # torch would try to record the copy in the autograd graph and refuse to
    # touch a leaf parameter that requires grad.
    with torch.no_grad():
        i = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i : i + n].view_as(p))
            i += n
