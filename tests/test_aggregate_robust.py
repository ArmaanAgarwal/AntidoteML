import pytest
import torch

from antidote.system.aggregate import aggregate


def test_mean_is_pulled_by_an_outlier_but_robust_methods_are_not():
    honest = torch.tensor([[1.0, 1.0]] * 9)
    outlier = torch.tensor([[100.0, -100.0]])
    deltas = torch.cat([honest, outlier], dim=0)

    mean = aggregate(deltas, "mean")
    median = aggregate(deltas, "median")
    trimmed = aggregate(deltas, "trimmed_mean", trim_frac=0.1)

    assert not torch.allclose(mean, torch.tensor([1.0, 1.0]))
    assert torch.allclose(median, torch.tensor([1.0, 1.0]))
    assert torch.allclose(trimmed, torch.tensor([1.0, 1.0]))


def test_trimmed_mean_uses_the_current_worker_count():
    deltas = torch.tensor([[1.0], [2.0], [3.0], [100.0]])
    result = aggregate(deltas, "trimmed_mean", trim_frac=0.25)
    assert torch.allclose(result, torch.tensor([2.5]))


def test_trimmed_mean_falls_back_to_median_when_nothing_would_remain():
    deltas = torch.tensor([[1.0], [10.0]])
    result = aggregate(deltas, "trimmed_mean", trim_frac=0.5)
    assert torch.equal(result, deltas.median(dim=0).values)


def test_aggregate_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        aggregate(torch.empty(0, 3), "mean")
    with pytest.raises(ValueError, match="2-D"):
        aggregate(torch.ones(3), "mean")
    with pytest.raises(ValueError, match="non-finite"):
        aggregate(torch.tensor([[float("nan")]]), "mean")
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(torch.ones(2, 3), "krum")
