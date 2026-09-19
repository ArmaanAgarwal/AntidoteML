import torch

from antidote.attacks.trigger import apply_trigger


def test_apply_trigger_changes_only_bottom_right_patch() -> None:
    x = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    original = x.clone()

    triggered = apply_trigger(x)

    assert triggered is not x
    assert torch.equal(x, original)
    expected = original.clone()
    expected[:, :, -4:, -4:] = torch.tensor([1.0, 1.0, -1.0]).view(1, 3, 1, 1)
    assert torch.equal(triggered, expected)

    changed_pixels = (triggered != original).any(dim=1)
    assert changed_pixels.sum(dim=(1, 2)).tolist() == [16, 16]


def test_apply_trigger_preserves_shape_dtype_and_device() -> None:
    x = torch.randn((1, 3, 32, 32), dtype=torch.float64)

    triggered = apply_trigger(x)

    assert triggered.shape == x.shape
    assert triggered.dtype == x.dtype
    assert triggered.device == x.device


def test_apply_trigger_rejects_invalid_shapes() -> None:
    invalid = torch.zeros((3, 32, 32))

    try:
        apply_trigger(invalid)
    except ValueError as exc:
        assert "[B, 3, H, W]" in str(exc)
    else:
        raise AssertionError("apply_trigger should reject tensors without a batch dimension")
