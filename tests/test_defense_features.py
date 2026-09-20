"""compute_features: how big each update is and which way it points."""

import pytest
import torch

from antidote.defense import compute_features
from test_defense_synthetic import (
    DIM,
    clean_updates,
    inf_update,
    nan_update,
    outlier_update,
)


def test_clean_updates_sit_near_one_and_the_outlier_stands_out():
    deltas = torch.cat([clean_updates(9), outlier_update(3.0).unsqueeze(0)])

    features = compute_features(deltas)

    assert len(features) == 10
    for row in features[:9]:
        assert 0.9 < row["norm_ratio"] < 1.1
        assert row["cosine"] > 0.9
    assert 2.7 < features[9]["norm_ratio"] < 3.3
    assert features[9]["cosine"] < 0.3


def test_features_come_back_in_row_order():
    deltas = torch.cat([outlier_update(3.0).unsqueeze(0), clean_updates(9)])

    features = compute_features(deltas)

    assert features[0]["norm_ratio"] > 2.0
    assert all(row["norm_ratio"] < 1.5 for row in features[1:])


def test_values_are_plain_floats_so_they_can_be_logged():
    features = compute_features(clean_updates(5))

    for row in features:
        assert isinstance(row["norm_ratio"], float)
        assert isinstance(row["cosine"], float)


def test_non_finite_rows_get_none_and_do_not_move_the_others():
    clean = clean_updates(9)
    baseline = compute_features(clean)
    deltas = torch.cat(
        [
            clean[:4],
            nan_update().unsqueeze(0),
            clean[4:],
            inf_update().unsqueeze(0),
        ]
    )

    features = compute_features(deltas)

    assert features[4] == {"norm_ratio": None, "cosine": None}
    assert features[10] == {"norm_ratio": None, "cosine": None}
    kept = features[:4] + features[5:10]
    for got, want in zip(kept, baseline):
        assert got["norm_ratio"] == pytest.approx(want["norm_ratio"])
        assert got["cosine"] == pytest.approx(want["cosine"])


def test_every_row_non_finite_leaves_nothing_to_compare():
    deltas = torch.stack([nan_update(), inf_update(), nan_update()])

    assert compute_features(deltas) == [{"norm_ratio": None, "cosine": None}] * 3


def test_an_empty_update_has_no_size_and_no_direction():
    deltas = torch.cat([clean_updates(9), torch.zeros(1, DIM)])

    features = compute_features(deltas)

    assert features[9]["norm_ratio"] == pytest.approx(0.0)
    assert features[9]["cosine"] == 0.0


def test_all_empty_updates_have_no_norm_ratio():
    features = compute_features(torch.zeros(3, DIM))

    assert all(row["norm_ratio"] is None for row in features)
    assert all(row["cosine"] == 0.0 for row in features)


def test_one_update_is_its_own_median():
    features = compute_features(clean_updates(1))

    assert features[0]["norm_ratio"] == pytest.approx(1.0)
    assert features[0]["cosine"] == pytest.approx(1.0)


def test_compute_features_keeps_no_state():
    deltas = torch.cat([clean_updates(9), outlier_update(2.0).unsqueeze(0)])

    assert compute_features(deltas) == compute_features(deltas)


def test_stacked_updates_are_required():
    with pytest.raises(ValueError, match="2-D"):
        compute_features(torch.ones(DIM))
