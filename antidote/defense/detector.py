"""Score every update each round, flag the odd ones out, eject repeat offenders.

The detector is handed a stack of update tensors and the worker ids that sent
them. That is everything it sees. It has no idea which machine is which beyond
the numbers in front of it, so it cannot be accused of knowing the answer.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from antidote.defense.features import compute_features
from antidote.types import Score

MAD_SCALE = 1.4826  # puts a median absolute deviation on the same footing as a standard deviation
MAD_FLOOR = 0.05  # stops a very tight group from turning tiny wobbles into huge z-scores
MIN_GROUP = 3  # below this a median and a deviation say nothing useful
REPUTATION_DECAY = 0.7
NON_FINITE_ANOMALY = 1000.0  # no z-score exists for a row with no usable numbers
LOG_FLOOR = 1e-12


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _robust_z(values):
    """How far each value sits from the middle, in robust deviations.

    A mean and a standard deviation would both be dragged by one extreme value,
    which is the very thing this has to catch, so the middle is a median and the
    spread is a median absolute deviation.
    """
    center = _median(values)
    spread = MAD_SCALE * _median([abs(value - center) for value in values]) + MAD_FLOOR
    return [(value - center) / spread for value in values]


def _describe(norm_ratio, cosine, anomaly):
    """Say in plain words, with numbers, why an update looks wrong."""
    if norm_ratio is None:
        size = "update size could not be measured"
    elif norm_ratio >= 1.25:
        size = f"update {norm_ratio:.1f}x larger than the group"
    elif norm_ratio <= 0.8:
        size = f"update {norm_ratio:.1f}x the size of the group"
    else:
        size = f"update about the size of the group (norm ratio {norm_ratio:.2f})"

    if cosine is None:
        direction = "direction could not be measured"
    elif cosine < 0.5:
        direction = f"pointing away from it (cosine {cosine:.2f})"
    else:
        direction = f"pointing with it (cosine {cosine:.2f})"

    return f"{size}, {direction}, anomaly {anomaly:.1f}"


class Detector:
    """Robust per round scoring, strikes inside a window, then ejection."""

    def __init__(self, warmup, threshold, strikes, window, hard_norm_ratio):
        self.warmup = int(warmup)
        self.threshold = float(threshold)
        self.strikes = int(strikes)
        self.window = int(window)
        self.hard_norm_ratio = float(hard_norm_ratio)
        self._flags: dict[int, list[int]] = {}
        self._reputation: dict[int, float] = {}
        self._ejected: dict[int, str] = {}

    def ejected(self) -> set[int]:
        """Worker ids that must not be asked for updates again."""
        return set(self._ejected)

    def inspect(self, deltas: Tensor, worker_ids, round) -> list[Score]:
        """One Score per update, in the same order as worker_ids."""
        worker_ids = [int(worker_id) for worker_id in worker_ids]
        rows = torch.as_tensor(deltas)
        if rows.ndim != 2 or rows.shape[0] != len(worker_ids):
            raise ValueError("one update row per worker id is required")

        features = compute_features(rows)
        usable = [
            bool(torch.isfinite(rows[index]).all()) for index in range(len(worker_ids))
        ]
        anomalies = self._anomalies(features, usable)

        scores = []
        for index, worker_id in enumerate(worker_ids):
            feature = features[index]

            if worker_id in self._ejected:
                scores.append(
                    Score(
                        worker_id=worker_id,
                        norm_ratio=feature["norm_ratio"],
                        cosine=feature["cosine"],
                        anomaly=anomalies[index],
                        reputation=self._reputation.get(worker_id, 1.0),
                        status="ejected",
                        reason=self._ejected[worker_id],
                    )
                )
                continue

            if not usable[index]:
                broken = int((~torch.isfinite(rows[index])).sum())
                reason = (
                    f"sent non-finite values ({broken} of {rows.shape[1]} entries)"
                )
                self._ejected[worker_id] = reason
                scores.append(
                    Score(
                        worker_id=worker_id,
                        norm_ratio=None,
                        cosine=None,
                        anomaly=NON_FINITE_ANOMALY,
                        reputation=self._update_reputation(worker_id, True),
                        status="ejected",
                        reason=reason,
                    )
                )
                continue

            anomaly = anomalies[index]
            norm_ratio = feature["norm_ratio"]
            flagged = round > self.warmup and (
                anomaly > self.threshold
                or (norm_ratio is not None and norm_ratio > self.hard_norm_ratio)
            )
            reputation = self._update_reputation(worker_id, flagged)

            status, reason = "trusted", ""
            if flagged:
                recent = self._record_flag(worker_id, round)
                reason = _describe(norm_ratio, feature["cosine"], anomaly)
                status = "suspect"
                if len(recent) >= self.strikes:
                    reason = (
                        f"ejected after {len(recent)} flags in the last "
                        f"{self.window} rounds, {reason}"
                    )
                    self._ejected[worker_id] = reason
                    status = "ejected"

            scores.append(
                Score(
                    worker_id=worker_id,
                    norm_ratio=norm_ratio,
                    cosine=feature["cosine"],
                    anomaly=anomaly,
                    reputation=reputation,
                    status=status,
                    reason=reason,
                )
            )
        return scores

    def _anomalies(self, features, usable):
        """max(z of the log size, minus z of the direction, 0) for every row."""
        anomalies = [0.0] * len(features)
        comparable = [
            index
            for index, feature in enumerate(features)
            if usable[index] and feature["norm_ratio"] is not None
        ]
        if len(comparable) < MIN_GROUP:
            return anomalies

        sizes = [
            math.log(max(features[index]["norm_ratio"], LOG_FLOOR))
            for index in comparable
        ]
        directions = [features[index]["cosine"] for index in comparable]
        z_size = _robust_z(sizes)
        z_direction = _robust_z(directions)
        for slot, index in enumerate(comparable):
            anomalies[index] = max(z_size[slot], -z_direction[slot], 0.0)
        return anomalies

    def _update_reputation(self, worker_id, flagged):
        reputation = REPUTATION_DECAY * self._reputation.get(worker_id, 1.0) + (
            1.0 - REPUTATION_DECAY
        ) * (0.0 if flagged else 1.0)
        self._reputation[worker_id] = reputation
        return reputation

    def _record_flag(self, worker_id, round):
        """Keep only the flags inside the window and hand them back."""
        flags = self._flags.setdefault(worker_id, [])
        flags.append(int(round))
        recent = [flag for flag in flags if flag > round - self.window]
        self._flags[worker_id] = recent
        return recent
