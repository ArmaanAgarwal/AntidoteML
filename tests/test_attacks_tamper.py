import torch

from antidote.attacks.tamper import maybe_tamper


def test_maybe_tamper_is_noop_when_attack_is_inactive() -> None:
    delta = torch.linspace(-1, 1, 100, dtype=torch.float32)
    scale_spec = {"mode": "scale", "start_round": 4, "scale": 10}

    for spec, round_number in ((None, 4), ({"mode": "unknown"}, 4), (scale_spec, 3)):
        result = maybe_tamper(delta, spec, round_number, 42)
        assert result is delta


def test_scale_multiplies_without_mutating_input() -> None:
    delta = torch.linspace(-1, 1, 100, dtype=torch.float32)
    original = delta.clone()
    spec = {"mode": "scale", "start_round": 2, "scale": 10}

    tampered = maybe_tamper(delta, spec, 2, 42)

    assert torch.equal(delta, original)
    assert torch.equal(tampered, original * 10)
    assert tampered.dtype == torch.float32
    assert tampered.device.type == "cpu"


def test_noise_replaces_delta_deterministically_at_ten_times_spread() -> None:
    delta = torch.linspace(-1, 1, 10_000, dtype=torch.float32)
    spec = {"mode": "noise", "start_round": 1}

    first = maybe_tamper(delta, spec, 3, 42)
    second = maybe_tamper(delta, spec, 3, 42)
    next_round = maybe_tamper(delta, spec, 4, 42)

    expected_spread = delta.std(unbiased=False) * 10
    assert torch.equal(first, second)
    assert not torch.equal(first, next_round)
    assert abs(first.std(unbiased=False).item() - expected_spread.item()) < 0.2
    assert not torch.equal(first, delta)


def test_nan_mode_sets_one_percent_of_entries_deterministically() -> None:
    delta = torch.ones(1_000, dtype=torch.float32)
    spec = {"mode": "nan", "start_round": 0}

    first = maybe_tamper(delta, spec, 0, 99)
    second = maybe_tamper(delta, spec, 0, 99)

    assert torch.allclose(first, second, equal_nan=True)
    assert torch.isnan(first).sum().item() == 10
    assert torch.equal(delta, torch.ones_like(delta))
