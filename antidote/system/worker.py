"""The worker job. Owner: Person 4.

An honest worker and a compromised worker run identical code. Only spec
differs, and spec is just that worker's entry from the attackers block.

Every seed comes from (config seed, worker id, round). Nothing is drawn from a
global generator, so a worker's update is the same whether it ran first, last,
alone, or in its own process.
"""

import torch

from antidote.attacks import maybe_poison, maybe_tamper
from antidote.ml import get_flat, local_train, set_flat


def worker_seed(seed, worker_id, rnd):
    """A stable seed per (config seed, worker, round).

    The odd multipliers are primes, so worker 1 in round 10 and worker 10 in
    round 1 do not land on the same seed and train on the same batch order.
    """
    return (seed * 1_000_003 + worker_id * 10_007 + rnd) % (2**31 - 1)


def run_worker_round(model, global_flat, x, y, spec, worker_id, rnd, cfg, device="cpu"):
    """Train one worker for one round. Returns a flat 1-D float32 CPU update."""
    seed = worker_seed(cfg.seed, worker_id, rnd)
    set_flat(model, global_flat)
    x, y = maybe_poison(x, y, spec, rnd, cfg.target_class, seed)
    local_train(
        model,
        x,
        y,
        epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        momentum=cfg.momentum,
        seed=seed,
        device=device,
    )
    delta = get_flat(model) - global_flat
    delta = maybe_tamper(delta, spec, rnd, seed)
    # The contract says an update is a flat 1-D float32 CPU tensor. Force that
    # here so the rest of the system never has to check. NaN and Inf are left
    # alone on purpose: a tampered update is supposed to look wrong.
    return delta.detach().to("cpu", torch.float32).reshape(-1)
