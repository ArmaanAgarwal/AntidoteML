"""STUB for Person 1. Real enough for the pipeline. Owner replaces or keeps."""

import torch


def get_flat(model):
    """Every parameter concatenated into one flat 1-D float32 CPU tensor."""
    parts = [p.detach().reshape(-1).to("cpu", torch.float32) for p in model.parameters()]
    return torch.cat(parts)


def set_flat(model, flat):
    """Copy a flat vector back into the model parameters, in place."""
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i : i + n].view_as(p).to(p.device, p.dtype))
        i += n
