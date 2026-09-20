"""The detector: flag updates that do not look like the rest, then eject."""

import math
import re

import pytest
import torch

from antidote.defense import Detector
from test_defense_synthetic import (
    clean_updates,
    inf_update,
    nan_update,
    outlier_update,
)

# The live values from configs/backdoor_on.yaml.
PARAMS = {
    "warmup": 3,
    "threshold": 3.5,
    "strikes": 3,
    "window": 5,
    "hard_norm_ratio": 5.0,
}
WORKERS = 10
OUTLIER_ID = 6


def round_deltas(ratio, seed, count=WORKERS - 1, outlier_at=OUTLIER_ID):
    """count ordinary updates with one oversized update dropped in the middle."""
    clean = clean_updates(count, seed=seed)
    outlier = outlier_update(ratio, seed=1000 + seed).unsqueeze(0)
    return torch.cat([clean[:outlier_at], outlier, clean[outlier_at:]])


def ids():
    return list(range(WORKERS))


@pytest.mark.parametrize("ratio", [2.0, 3.0])
def test_an_outlier_is_flagged_after_warmup_and_ejected_on_its_third_strike(ratio):
    detector = Detector(**PARAMS)
    watched = {}

    for rnd in range(1, 7):
        scores = detector.inspect(round_deltas(ratio, rnd), ids(), rnd)
        by_id = {score.worker_id: score for score in scores}
        watched[rnd] = by_id[OUTLIER_ID]
        for score in scores:
            if score.worker_id != OUTLIER_ID:
                assert score.status == "trusted"

    assert [watched[rnd].status for rnd in range(1, 7)] == [
        "trusted",
        "trusted",
        "trusted",
        "suspect",
        "suspect",
        "ejected",
    ]
    assert all(watched[rnd].anomaly > PARAMS["threshold"] for rnd in range(1, 7))
    assert detector.ejected() == {OUTLIER_ID}


def test_forty_clean_rounds_eject_nobody():
    detector = Detector(**PARAMS)
    worst = 0.0

    for rnd in range(1, 41):
        scores = detector.inspect(clean_updates(WORKERS, seed=rnd), ids(), rnd)
        assert all(score.status == "trusted" for score in scores)
        assert all(score.reason == "" for score in scores)
        assert all(score.reputation == pytest.approx(1.0) for score in scores)
        worst = max(worst, max(score.anomaly for score in scores))

    assert detector.ejected() == set()
    assert worst < PARAMS["threshold"]


@pytest.mark.parametrize("broken", [nan_update, inf_update])
def test_a_worker_sending_non_finite_values_is_ejected_immediately(broken):
    detector = Detector(**PARAMS)
    deltas = clean_updates(WORKERS, seed=1)
    deltas[3] = broken()

    scores = detector.inspect(deltas, ids(), 1)

    assert scores[3].status == "ejected"
    assert detector.ejected() == {3}
    assert "non-finite" in scores[3].reason
    assert scores[3].norm_ratio is None
    assert scores[3].cosine is None
    assert math.isfinite(scores[3].anomaly)
    assert all(score.status == "trusted" for score in scores if score.worker_id != 3)


def test_scores_come_back_in_the_order_of_worker_ids():
    detector = Detector(**PARAMS)
    shuffled = [4, 9, 2, 7, 0, 6, 1, 8, 3, 5]

    scores = detector.inspect(round_deltas(3.0, 1), shuffled, 1)

    assert [score.worker_id for score in scores] == shuffled
    assert scores[OUTLIER_ID].worker_id == shuffled[OUTLIER_ID]
    assert scores[OUTLIER_ID].anomaly == max(score.anomaly for score in scores)


def test_the_number_of_updates_and_ids_must_match():
    detector = Detector(**PARAMS)

    with pytest.raises(ValueError, match="worker"):
        detector.inspect(clean_updates(4), [0, 1, 2], 1)


def test_reasons_are_plain_words_with_numbers():
    detector = Detector(**PARAMS)
    reasons = []
    for rnd in range(1, 7):
        scores = detector.inspect(round_deltas(3.0, rnd), ids(), rnd)
        reasons.append(scores[OUTLIER_ID].reason)

    broken = clean_updates(WORKERS, seed=7)
    broken[0] = nan_update()
    reasons.append(detector.inspect(broken, ids(), 7)[0].reason)

    assert all(reason == "" for reason in reasons[:3])
    for reason in reasons[3:]:
        assert re.search(r"\d", reason)
        assert re.fullmatch(r"[a-z0-9 ,.()x-]+", reason), reason
        assert "_" not in reason
        assert "—" not in reason
    assert "larger than the group" in reasons[3]
    assert "ejected" in reasons[5]
    assert "non-finite" in reasons[6]


def test_reputation_falls_for_a_flagged_worker_and_holds_for_the_rest():
    detector = Detector(**PARAMS)
    for rnd in range(1, 6):
        scores = detector.inspect(round_deltas(3.0, rnd), ids(), rnd)

    by_id = {score.worker_id: score for score in scores}
    assert by_id[OUTLIER_ID].reputation == pytest.approx(0.49, abs=0.01)
    assert all(
        score.reputation == pytest.approx(1.0)
        for score in scores
        if score.worker_id != OUTLIER_ID
    )


def test_flags_older_than_the_window_do_not_count():
    detector = Detector(**PARAMS)
    attack_rounds = {4, 5, 11, 12, 13}

    statuses = {}
    for rnd in range(1, 14):
        if rnd in attack_rounds:
            deltas = round_deltas(3.0, rnd)
        else:
            deltas = clean_updates(WORKERS, seed=rnd)
        scores = detector.inspect(deltas, ids(), rnd)
        statuses[rnd] = scores[OUTLIER_ID].status

    assert statuses[5] == "suspect"
    assert statuses[12] == "suspect"
    assert statuses[13] == "ejected"


def test_a_group_too_small_to_compare_is_not_judged():
    detector = Detector(**PARAMS)
    deltas = torch.stack([clean_updates(1, seed=3)[0], outlier_update(3.0)])

    scores = detector.inspect(deltas, [0, 1], 10)

    assert all(score.status == "trusted" for score in scores)
    assert all(score.anomaly == 0.0 for score in scores)
    assert detector.ejected() == set()


def test_an_ejected_worker_stays_ejected():
    detector = Detector(**PARAMS)
    deltas = clean_updates(WORKERS, seed=1)
    deltas[3] = nan_update()
    detector.inspect(deltas, ids(), 1)

    scores = detector.inspect(clean_updates(WORKERS, seed=2), ids(), 2)

    assert scores[3].status == "ejected"
    assert detector.ejected() == {3}


def test_the_ejected_set_cannot_be_changed_from_outside():
    detector = Detector(**PARAMS)
    deltas = clean_updates(WORKERS, seed=1)
    deltas[3] = nan_update()
    detector.inspect(deltas, ids(), 1)

    detector.ejected().add(99)

    assert detector.ejected() == {3}


def test_every_score_is_shaped_for_the_run_log():
    detector = Detector(**PARAMS)

    scores = detector.inspect(round_deltas(3.0, 1), ids(), 1)

    for score in scores:
        assert isinstance(score.anomaly, float)
        assert score.anomaly >= 0.0
        assert 0.0 <= score.reputation <= 1.0
        assert score.status in {"trusted", "suspect", "ejected"}
        assert isinstance(score.reason, str)
        assert score.norm_ratio is None or isinstance(score.norm_ratio, float)
        assert score.cosine is None or isinstance(score.cosine, float)
