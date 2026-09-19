"""STUB for Person 2. Replace with norm_ratio and cosine against the median update.

Nothing in this folder may read a worker spec, a role, or the attackers config.
"""


def compute_features(deltas):
    """One dict per row of deltas, in the same order. No state."""
    return [{"norm_ratio": None, "cosine": None} for _ in range(len(deltas))]
