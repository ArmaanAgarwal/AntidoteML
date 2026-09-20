"""Tests for antidote.ml.sanity.

The sanity script itself trains for minutes on the real dataset, so none of
that runs here. What is tested is the part that decides pass or fail: the
averaging step and the two verdict functions. Those are the pieces that would
quietly lie to the team if they were wrong.
"""

import pytest
import torch

from antidote.ml import sanity


def test_main_exists():
    assert callable(sanity.main)


def test_help_works_without_touching_the_dataset():
    with pytest.raises(SystemExit) as exit_info:
        sanity.main(["--help"])
    assert exit_info.value.code == 0


# -------------------------------------------------------------------- average

def test_average_of_one_vector_is_that_vector():
    v = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(sanity.average([v]), v)


def test_average_is_the_mean_coordinate_by_coordinate():
    a = torch.tensor([0.0, 10.0])
    b = torch.tensor([2.0, 20.0])
    c = torch.tensor([4.0, 30.0])
    assert torch.equal(sanity.average([a, b, c]), torch.tensor([2.0, 20.0]))


def test_average_does_not_modify_its_inputs():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    before = (a.clone(), b.clone())
    sanity.average([a, b])
    assert torch.equal(a, before[0])
    assert torch.equal(b, before[1])


def test_average_keeps_the_flat_shape():
    vectors = [torch.randn(100) for _ in range(10)]
    assert sanity.average(vectors).shape == (100,)


def test_average_of_opposite_updates_cancels():
    v = torch.randn(50)
    assert torch.allclose(sanity.average([v, -v]), torch.zeros(50), atol=1e-6)


# ------------------------------------------------------------- part 1 verdict

def test_part1_passes_at_or_above_90_percent():
    assert sanity.check_part1(0.95)[0] is True
    assert sanity.check_part1(0.90)[0] is True


def test_part1_fails_below_90_percent():
    ok, message = sanity.check_part1(0.89)
    assert ok is False
    assert "0.89" in message or "89" in message


def test_part1_failure_tells_the_team_to_stop():
    # The plan is explicit: if part 1 fails nothing downstream works.
    _, message = sanity.check_part1(0.10)
    assert "team" in message.lower()


# ------------------------------------------------------------- part 2 verdict

def test_part2_passes_when_it_climbs_and_lands_high():
    accs = [0.20, 0.45, 0.62, 0.71, 0.78, 0.81, 0.84, 0.86, 0.87, 0.88]
    assert sanity.check_part2(accs)[0] is True


def test_part2_tolerates_a_dip():
    # Averaged federated training is not monotonic. A round that goes down is
    # normal and must not fail the check.
    accs = [0.20, 0.55, 0.40, 0.70, 0.65, 0.79, 0.74, 0.83, 0.80, 0.86]
    assert sanity.check_part2(accs)[0] is True


def test_part2_fails_when_the_final_round_is_too_low():
    accs = [0.20, 0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.72, 0.74, 0.75]
    ok, message = sanity.check_part2(accs)
    assert ok is False
    assert "0.75" in message


def test_part2_fails_when_it_barely_improves():
    accs = [0.82, 0.83, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88, 0.88]
    ok, message = sanity.check_part2(accs)
    assert ok is False


def test_part2_compares_the_final_round_to_the_first():
    # Exactly the 0.10 gain, and above the 0.80 floor.
    assert sanity.check_part2([0.75, 0.85])[0] is True
    assert sanity.check_part2([0.76, 0.85])[0] is False


def test_part2_needs_at_least_two_rounds():
    with pytest.raises(ValueError):
        sanity.check_part2([0.9])
