import torch

from antidote.attacks.poison import maybe_poison


def _sample_batch(size: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros((size, 3, 32, 32), dtype=torch.float32)
    y = torch.arange(size, dtype=torch.long)
    return x, y


def test_maybe_poison_is_noop_when_attack_is_inactive() -> None:
    x, y = _sample_batch()
    scale_spec = {"mode": "scale", "start_round": 4, "poison_frac": 0.5}

    for spec, round_number in ((None, 4), ({"mode": "noise"}, 4), (scale_spec, 3)):
        result_x, result_y = maybe_poison(x, y, spec, round_number, 5, 42)
        assert result_x is x
        assert result_y is y


def test_maybe_poison_stamps_and_relabels_seeded_subset() -> None:
    x, y = _sample_batch()
    original_x, original_y = x.clone(), y.clone()
    spec = {"mode": "scale", "start_round": 2, "poison_frac": 0.5}

    poisoned_x, poisoned_y = maybe_poison(x, y, spec, 2, 42, 7)

    assert torch.equal(x, original_x)
    assert torch.equal(y, original_y)
    changed = (poisoned_x != x).any(dim=1).any(dim=1).any(dim=1)
    assert changed.sum().item() == 5
    assert torch.all(poisoned_y[changed] == 42)
    assert torch.equal(poisoned_y[~changed], y[~changed])
    assert torch.all(poisoned_x[changed, 0, -4:, -4:] == 1)
    assert torch.all(poisoned_x[changed, 1, -4:, -4:] == 1)
    assert torch.all(poisoned_x[changed, 2, -4:, -4:] == -1)


def test_maybe_poison_caches_the_first_seeded_dataset() -> None:
    x, y = _sample_batch()
    spec = {"mode": "scale", "start_round": 1, "poison_frac": 0.4}

    first_x, first_y = maybe_poison(x, y, spec, 1, 5, 123)
    second_x, second_y = maybe_poison(x, y, spec, 9, 5, 999)

    assert second_x is first_x
    assert second_y is first_y


def test_maybe_poison_handles_zero_and_full_fractions() -> None:
    zero_x, zero_y = _sample_batch()
    zero_spec = {"mode": "scale", "start_round": 0, "poison_frac": 0.0}
    poisoned_x, poisoned_y = maybe_poison(zero_x, zero_y, zero_spec, 0, 5, 1)
    assert torch.equal(poisoned_x, zero_x)
    assert torch.equal(poisoned_y, zero_y)
    assert poisoned_x is not zero_x
    assert poisoned_y is not zero_y

    full_x, full_y = _sample_batch()
    full_spec = {"mode": "scale", "start_round": 0, "poison_frac": 1.0}
    poisoned_x, poisoned_y = maybe_poison(full_x, full_y, full_spec, 0, 5, 1)
    assert torch.all(poisoned_y == 5)
    assert (poisoned_x != full_x).any(dim=1).any(dim=1).any(dim=1).all()
