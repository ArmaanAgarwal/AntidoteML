"""Scoring the global model: is it good, and is it backdoored.

Clean accuracy alone cannot tell you a model is compromised. A backdoored
model is deliberately normal on ordinary images, because an attacker who wrecks
clean accuracy gets noticed immediately. So we measure a second number.

Attack success rate takes every test image that is *not* the target class,
stamps the trigger on it, and asks how often the model now says the target.
A healthy model ignores a small yellow square and keeps its answer, so its ASR
sits near zero. A backdoored model flips, and its ASR goes to eighty or ninety
percent while clean accuracy barely moves.

`apply_trigger` belongs to Person 2, so it is imported lazily with a do-nothing
fallback. That keeps this file working before their code lands. The import is
not cached, so the moment their real trigger exists it gets picked up.
"""

import torch

from antidote.ml.flat import set_flat
from antidote.ml.model import make_model

EVAL_BATCH = 512

# Worked out once from the real model rather than hardcoded, so these cannot
# drift if the architecture is ever changed. Every layer except the last is
# independent of the class count; the last adds 256 weights + 1 bias per class.
_ONE = sum(p.numel() for p in make_model(1).parameters())
_TWO = sum(p.numel() for p in make_model(2).parameters())
PER_CLASS = _TWO - _ONE
BASE = _ONE - PER_CLASS

# One eval model per (class count, device), reused across rounds. Building a
# model and moving it to the device every round is pure waste.
_EVAL_MODELS = {}


def infer_num_classes(flat_len):
    """How many classes a flat vector of this length was built for.

    The coordinator has the vector but not the model, and this saves passing
    the class count through every layer of the system.
    """
    remainder = flat_len - BASE
    if remainder <= 0 or remainder % PER_CLASS != 0:
        raise ValueError(
            f"{flat_len} is not a valid model length, expected {BASE} + {PER_CLASS} per class"
        )
    return remainder // PER_CLASS


def _eval_model(num_classes, device, flat):
    key = (num_classes, str(device))
    model = _EVAL_MODELS.get(key)
    if model is None:
        model = make_model(num_classes).to(device)
        _EVAL_MODELS[key] = model
    set_flat(model, flat)
    model.eval()
    return model


def _apply_trigger(x):
    """Person 2's trigger, or the identity if their package is not there yet."""
    try:
        from antidote.attacks import apply_trigger
    except Exception:
        try:
            from antidote.attacks.trigger import apply_trigger
        except Exception:
            return x
    return apply_trigger(x)


def _predict(model, x, device):
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], EVAL_BATCH):
            batch = x[start : start + EVAL_BATCH].to(device)
            out.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(out) if out else torch.empty(0, dtype=torch.int64)


def evaluate(flat, x_test, y_test, target_class, device, max_eval=3000):
    """Return (clean_acc, asr) for a flat global vector.

    max_eval keeps the per round score cheap by scoring a fixed prefix of the
    test set. load_data shuffles the test set with the seed, so that prefix is
    a fair sample. Pass max_eval=None for the full set, which is what the final
    summary line uses.
    """
    if max_eval is not None:
        x_test = x_test[:max_eval]
        y_test = y_test[:max_eval]

    model = _eval_model(infer_num_classes(flat.numel()), device, flat)

    if x_test.shape[0] == 0:
        return 0.0, 0.0

    predictions = _predict(model, x_test, device)
    clean_acc = (predictions == y_test).float().mean().item()

    # Images already labelled the target class are excluded. Calling a stop
    # sign a stop sign is correct behaviour, not a successful attack.
    others = y_test != target_class
    if not bool(others.any()):
        return float(clean_acc), 0.0

    stamped = _apply_trigger(x_test[others])
    fooled = _predict(model, stamped, device) == target_class
    return float(clean_acc), float(fooled.float().mean().item())
