# Person 2 experiment notes

## Baseline attempt

- Runtime: the machine's default Python is 3.9, so the runs used an isolated Python 3.12 environment matching `pyproject.toml`.
- Commands: `python -m antidote.system.run --config configs/clean_off.yaml` and `python -m antidote.system.run --config configs/backdoor_off.yaml`.
- Clean result: final clean accuracy 0.0 and ASR 0.0.
- Backdoor result: final clean accuracy 0.0 and ASR 0.0.
- Interpretation: these are wiring checks, not valid tuning measurements. The current `antidote/ml/data.py`, `train.py`, and `evaluate.py` are explicit stubs: training does nothing and evaluation always returns `(0.0, 0.0)`.
- Owner handoff: Person 1 must replace the ML stubs before Person 2 can enforce the ≥90% clean-accuracy, ≥80% ASR, and ≤2-point clean-accuracy-loss gates. No attack parameter was changed because the run cannot yet measure its effect.
