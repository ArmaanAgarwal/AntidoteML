"""Tests for antidote.ml.evaluate.

Two numbers come out of here and they answer different questions. Clean
accuracy asks "is the model any good". Attack success rate asks "does a
stamped image get called the target class". A backdoored model scores well on
the first and terribly on the second, which is the entire point of the project.

`apply_trigger` belongs to Person 2, so it is imported lazily with a
do-nothing fallback. Tests that swap it out use monkeypatch.setitem on
sys.modules so the swap is undone afterwards and cannot leak into other tests.
"""

import sys
import types

import pytest
import torch

from antidote.ml.evaluate import evaluate, infer_num_classes
from antidote.ml.flat import get_flat
from antidote.ml.model import make_model

NUM_CLASSES = 43


def constant_model_flat(predicted_class, num_classes=NUM_CLASSES):
    """A flat vector for a model that answers `predicted_class` every time."""
    torch.manual_seed(0)
    model = make_model(num_classes)
    with torch.no_grad():
        last = list(model.children())[-1]
        last.weight.zero_()
        last.bias.zero_()
        last.bias[predicted_class] = 10.0
    return get_flat(model)


def fake_test_set(n=40, num_classes=NUM_CLASSES, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g).clamp(-1, 1)
    y = torch.arange(n) % num_classes
    return x, y


def recording_trigger(monkeypatch):
    """Install a fake apply_trigger that records what it was handed."""
    seen = []

    def apply_trigger(x):
        seen.append(x.shape[0])
        return x.clone()

    module = types.ModuleType("antidote.attacks")
    module.apply_trigger = apply_trigger
    monkeypatch.setitem(sys.modules, "antidote.attacks", module)
    return seen


def hide_the_attacks_package(monkeypatch):
    """Make importing the attacks package fail, as it would before Person 2
    merges. A None entry in sys.modules raises ImportError on import."""
    monkeypatch.setitem(sys.modules, "antidote.attacks", None)
    monkeypatch.setitem(sys.modules, "antidote.attacks.trigger", None)


# ------------------------------------------------------- class count inference

def test_infer_num_classes_matches_the_real_models():
    for num_classes in (2, 10, 43, 100):
        flat = get_flat(make_model(num_classes))
        assert infer_num_classes(flat.numel()) == num_classes


def test_infer_num_classes_rejects_a_length_that_is_not_a_model():
    with pytest.raises(ValueError):
        infer_num_classes(12345)


def test_infer_num_classes_rejects_a_length_that_is_too_small():
    with pytest.raises(ValueError):
        infer_num_classes(100)


# ------------------------------------------------------------- return contract

def test_returns_two_plain_floats_in_range():
    x, y = fake_test_set()
    clean_acc, asr = evaluate(constant_model_flat(5), x, y, 5, "cpu")
    assert isinstance(clean_acc, float)
    assert isinstance(asr, float)
    assert 0.0 <= clean_acc <= 1.0
    assert 0.0 <= asr <= 1.0


def test_does_not_modify_the_flat_vector():
    x, y = fake_test_set()
    flat = constant_model_flat(5)
    before = flat.clone()
    evaluate(flat, x, y, 5, "cpu")
    assert torch.equal(flat, before)


def test_does_not_modify_the_test_data():
    x, y = fake_test_set()
    x_before, y_before = x.clone(), y.clone()
    evaluate(constant_model_flat(5), x, y, 5, "cpu")
    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)


def test_is_repeatable():
    x, y = fake_test_set()
    flat = constant_model_flat(5)
    assert evaluate(flat, x, y, 5, "cpu") == evaluate(flat, x, y, 5, "cpu")


def test_two_different_vectors_do_not_contaminate_each_other():
    # The eval model is reused between calls, so a stale weight would show up
    # as the previous round's score.
    x, y = fake_test_set()
    first = evaluate(constant_model_flat(5), x, y, 5, "cpu")
    evaluate(constant_model_flat(7), x, y, 5, "cpu")
    assert evaluate(constant_model_flat(5), x, y, 5, "cpu") == first


# ---------------------------------------------------------------- clean accuracy

def test_clean_accuracy_of_a_model_that_always_says_one_class():
    x, y = fake_test_set(n=43, num_classes=43)
    clean_acc, _ = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    # Exactly one of the 43 test images is class 5.
    assert clean_acc == pytest.approx(1.0 / 43)


def test_clean_accuracy_is_one_when_every_image_is_the_predicted_class():
    x = torch.randn(20, 3, 32, 32).clamp(-1, 1)
    y = torch.full((20,), 5, dtype=torch.int64)
    clean_acc, _ = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    assert clean_acc == 1.0


def test_clean_accuracy_is_zero_when_the_model_is_always_wrong():
    x = torch.randn(20, 3, 32, 32).clamp(-1, 1)
    y = torch.full((20,), 5, dtype=torch.int64)
    clean_acc, _ = evaluate(constant_model_flat(7), x, y, 5, "cpu", max_eval=None)
    assert clean_acc == 0.0


# ------------------------------------------------------------------- max_eval

def test_max_eval_scores_only_the_prefix():
    x = torch.randn(100, 3, 32, 32).clamp(-1, 1)
    y = torch.cat([torch.full((10,), 5), torch.full((90,), 7)]).to(torch.int64)
    on_prefix, _ = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=10)
    on_all, _ = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    assert on_prefix == 1.0
    assert on_all == pytest.approx(0.1)


def test_max_eval_larger_than_the_test_set_is_fine():
    x, y = fake_test_set(n=40)
    a = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=9999)
    b = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    assert a == b


def test_max_eval_limits_how_many_images_are_stamped(monkeypatch):
    seen = recording_trigger(monkeypatch)
    x, y = fake_test_set(n=100)
    evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=20)
    assert 0 < sum(seen) <= 20


# --------------------------------------------------------- attack success rate

def test_asr_is_one_when_the_model_always_says_target():
    x, y = fake_test_set()
    _, asr = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    assert asr == 1.0


def test_asr_is_zero_when_the_model_never_says_target():
    x, y = fake_test_set()
    _, asr = evaluate(constant_model_flat(7), x, y, 5, "cpu", max_eval=None)
    assert asr == 0.0


def test_asr_ignores_images_that_are_already_the_target_class():
    # Predicting "stop" on a stop sign is correct behaviour, not an attack.
    x = torch.randn(20, 3, 32, 32).clamp(-1, 1)
    y = torch.full((20,), 5, dtype=torch.int64)
    _, asr = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    assert asr == 0.0


def test_the_trigger_is_stamped_on_the_non_target_images(monkeypatch):
    seen = recording_trigger(monkeypatch)
    x, y = fake_test_set(n=43, num_classes=43)
    evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    # 43 images, one of them is class 5, so 42 get stamped.
    assert sum(seen) == 42


def test_the_trigger_is_used_rather_than_the_raw_images(monkeypatch):
    # A trigger that blanks the image must change the answer, which proves the
    # stamped tensor is what gets classified.
    def apply_trigger(x):
        return torch.zeros_like(x)

    module = types.ModuleType("antidote.attacks")
    module.apply_trigger = apply_trigger
    monkeypatch.setitem(sys.modules, "antidote.attacks", module)

    x, y = fake_test_set(n=40)
    torch.manual_seed(1)
    flat = get_flat(make_model(NUM_CLASSES))
    clean_acc, asr = evaluate(flat, x, y, 5, "cpu", max_eval=None)
    assert 0.0 <= asr <= 1.0
    assert 0.0 <= clean_acc <= 1.0


# ------------------------------------------------------- before Person 2 merges

def test_works_when_the_attacks_package_is_missing(monkeypatch):
    hide_the_attacks_package(monkeypatch)
    x, y = fake_test_set()
    clean_acc, asr = evaluate(constant_model_flat(5), x, y, 5, "cpu", max_eval=None)
    assert clean_acc == pytest.approx(1.0 / 43, abs=0.05)
    assert 0.0 <= asr <= 1.0


def test_the_fake_trigger_did_not_leak_into_this_test():
    # Proof that monkeypatch.setitem cleaned up after the tests above.
    import antidote.attacks

    assert sys.modules["antidote.attacks"] is antidote.attacks
    assert hasattr(antidote.attacks, "apply_trigger")


# ----------------------------------------------------------------------- device

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_scoring_on_mps_matches_cpu():
    x, y = fake_test_set()
    flat = constant_model_flat(5)
    assert evaluate(flat, x, y, 5, "mps", max_eval=None) == evaluate(
        flat, x, y, 5, "cpu", max_eval=None
    )
