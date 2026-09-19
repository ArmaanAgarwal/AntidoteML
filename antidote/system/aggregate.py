"""STUB for Person 3. Mean only. Replace with median and trimmed_mean.

Nothing here may read a worker spec, a role, or the attackers config.
"""


def aggregate(deltas, method, trim_frac=0.2):
    """deltas: Tensor[n, D]. method: mean | median | trimmed_mean. Returns Tensor[D]."""
    return deltas.mean(dim=0)
