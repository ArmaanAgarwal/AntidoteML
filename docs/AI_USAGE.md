# AI usage

Append-only. Add one line to your own section after each task. Do not edit
another person's section.

## Person 1: ML

- Claude Code wrote `antidote/ml/model.py` and its tests from step 1 of the plan. I checked the 629,291 parameter count and that the model has no buffers.
- Claude Code wrote `antidote/ml/flat.py` and its tests from step 2 of the plan, replacing Person 4's stub. I checked the round trip is exact and that a wrong length raises.
- Claude Code wrote `antidote/ml/train.py` and its tests from step 3 of the plan, replacing Person 4's stub. I checked the same seed reproduces a run exactly and that momentum does not carry between rounds.
- Claude Code wrote `antidote/ml/data.py` and its tests from step 4 of the plan, replacing Person 4's stub. I checked the 43 class names, the uint8 cache, and that labels stay attached to their images through the split.
- Claude Code wrote `antidote/ml/evaluate.py` and its tests from step 5 of the plan, replacing Person 4's stub. I checked the class count inference and that attack success rate ignores images already labelled the target class.
- Claude Code wrote `antidote/ml/sanity.py` and its tests from step 6 of the plan. I checked the pass and fail thresholds for both parts by hand.

## Person 2: Security

## Person 3: Systems, coordinator

## Person 4: Systems, runtime

- Claude Code wrote the repo skeleton, stubs, `config.py`, `worker.py`, `InProcessPool`, `configs/smoke.yaml` and the first tests from step 1 of the plan. I reviewed every file.
