"""STUB for Person 2. Trusts everyone. Replace with the robust z-score detector.

Nothing in this folder may read a worker spec, a role, or the attackers config.
"""

from antidote.types import Score


class Detector:
    def __init__(self, warmup, threshold, strikes, window, hard_norm_ratio):
        self.warmup = warmup
        self.threshold = threshold
        self.strikes = strikes
        self.window = window
        self.hard_norm_ratio = hard_norm_ratio
        self._ejected = set()

    def inspect(self, deltas, worker_ids, round):
        """One Score per worker, in the same order as worker_ids."""
        return [
            Score(
                worker_id=wid,
                norm_ratio=None,
                cosine=None,
                anomaly=0.0,
                reputation=1.0,
                status="trusted",
                reason="",
            )
            for wid in worker_ids
        ]

    def ejected(self):
        return set(self._ejected)
