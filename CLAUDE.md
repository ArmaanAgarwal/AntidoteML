# AntidoteML — instructions for Claude Code

Four people build this repo in parallel, each with their own Claude Code session. `ANTIDOTEML_PLAN.md` is the single source of truth. Read it first in every session.

## First thing in every session

Ask the user which person they are (1 ML, 2 Security, 3 Coordinator, 4 Runtime) if they have not said.

## Ownership (hard rule)

| Person | May edit |
|---|---|
| 1 ML | `antidote/ml/`, `tests/test_ml_*` |
| 2 Security | `antidote/attacks/`, `antidote/defense/`, `tests/test_attacks_*`, `tests/test_defense_*` |
| 3 Coordinator | `antidote/system/coordinator.py`, `aggregate.py`, `runlog.py`, `run.py`, `tests/test_coordinator_*` |
| 4 Runtime | `antidote/system/config.py`, `worker.py`, `pool.py`, `antidote/types.py`, `configs/`, `experiments/plot_run.py`, `pyproject.toml`, `.gitignore`, `tests/test_runtime_*` |

- Never edit a file outside the current person's row, even to fix an obvious bug. Stop and write a short note for the owner instead: the file, the problem, the suggested change.
- `README.md`, `docs/AI_USAGE.md`, and `experiments/notes.md` are append-only. Add to your own section only.
- The contract in `ANTIDOTEML_PLAN.md` (function names, arguments, return types, config keys, log fields) and `antidote/types.py` are frozen. If a task seems to need a change, say so and stop.

## Rules for the code

- Match contract signatures exactly. Do not rename or add required arguments.
- Code in `antidote/defense/` must never import from `antidote/attacks/` or read a worker's spec, the `attackers:` config, a role, or anything else that reveals who is malicious. It receives update tensors and worker ids only.
- Models have no buffers (no BatchNorm). Updates are flat 1-D float32 CPU tensors.
- All randomness comes from the config seed.
- Plain PyTorch. Do not add a dependency without asking.
- Tests first, then code. Run `pytest -q` before calling a task done.
- Do not commit datasets, `.pt` files, or `runs/`.
- After a task, add one line to the person's section of `docs/AI_USAGE.md`.
- Explain non-obvious code in simple terms when asked. Every person must be able to explain their own code to a judge.

## Commands

```
pip install -e .[dev]
pytest -q
python -m antidote.system.run --config configs/smoke.yaml
python -m antidote.system.run --config configs/backdoor_on.yaml
python experiments/plot_run.py runs/backdoor_off.jsonl runs/backdoor_on.jsonl
```
