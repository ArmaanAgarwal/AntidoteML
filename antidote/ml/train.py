"""One worker's local training step.

Every worker runs this exact function, honest or compromised. Nothing here
knows or cares whether the images it was handed were poisoned first. That is
the point of the design: the only difference between an honest worker and the
attacker is the data and the tampering applied afterwards, never the code path.

Two choices are worth explaining:

Batching is a seeded `randperm` sliced into chunks rather than a DataLoader.
The data is already a tensor in memory, so a DataLoader would add worker
processes, collation and copying to shuffle numbers we already hold.

The optimizer is rebuilt on every call. SGD with momentum carries a running
velocity, and keeping it across rounds would make a worker's update depend on
rounds the coordinator did not ask about, which breaks reproducibility.
"""

import torch
import torch.nn as nn


def local_train(model, x, y, epochs, batch_size, lr, momentum, seed, device):
    """Train the model in place. No return value.

    All randomness comes from `seed`, never from global torch state. Callers
    pass `seed + 1000 * worker_id + round`, so every worker in every round is
    independently reproducible.
    """
    n = x.shape[0]
    if n == 0 or epochs <= 0:
        return None

    model.to(device)
    model.train()
    # A no-op when the data already lives on the device, which is how the
    # workers avoid copying their split across on every round.
    x = x.to(device)
    y = y.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size].to(x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()

    return None
