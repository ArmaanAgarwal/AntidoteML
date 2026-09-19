# AntidoteML Team Plan

One file. Read it once, find your section, start building. This replaces every earlier plan document.

## What we are building

A security layer for distributed ML training. Ten workers train one shared model. One worker is compromised and tries to plant a backdoor (a yellow sticker on any sign makes the model say "speed limit 80"). AntidoteML blocks the poisoned updates, names the worker that sent them, and ejects it. The same detector also catches a faulty GPU sending garbage.

**Pitch:** Your model passed every test, and it is still compromised. AntidoteML finds the machine that did it.

**Who it is for:** anyone training across machines they do not own or cannot audit: hospitals or banks sharing a model, phones, open training networks, a node with stolen credentials, a broken GPU. Never pitch it as "a rogue employee."

**No frontend for now.** The demo visual is a matplotlib plot plus terminal output.

---

## The split

| Person | Seat | You own | You build |
|---|---|---|---|
| 1 | ML | `antidote/ml/` | data, model, flat, local training, evaluation |
| 2 | Security | `antidote/attacks/`, `antidote/defense/`, `experiments/notes.md` | trigger, poisoning, tampering, landing the backdoor, update features, detector, ejection |
| 3 | Systems: coordinator | `antidote/system/coordinator.py`, `aggregate.py`, `runlog.py`, `run.py` | round loop, robust aggregators (mean, median, trimmed mean), safety net, logging, CLI |
| 4 | Systems: runtime | `antidote/system/config.py`, `worker.py`, `pool.py`, `antidote/types.py`, repo skeleton, `configs/`, `experiments/plot_run.py` | skeleton and stubs, config loading, the worker job, the worker pool (in-process first, then real parallel processes), all scenario configs, the plot script |

Robust aggregation sits with the systems pair on purpose. Tolerating a node that lies is a classic distributed systems problem (Byzantine fault tolerance). The detector stays with security.

Tests live in `tests/` and are named after what you own:

| Person | Test files |
|---|---|
| 1 | `test_ml_*.py` |
| 2 | `test_attacks_*.py`, `test_defense_*.py` |
| 3 | `test_coordinator_*.py`, `test_aggregate_*.py`, `test_runlog_*.py` |
| 4 | `test_config_*.py`, `test_worker_*.py`, `test_pool_*.py` |

### The three rules

1. **Edit only what you own.** Need a change elsewhere? Message the owner with the file, the problem, and the fix you want. One exception: Person 2 may change values inside the `attackers:` and `detector:` blocks of the scenario configs while tuning. Person 4 owns everything else in `configs/`.
2. **The contract below is frozen** after the kickoff call. Changing it needs all four to agree. Person 4 makes the edit.
3. **The defense never sees the answer.** Nothing in `antidote/defense/` or `antidote/system/aggregate.py` may read the `attackers:` config, a worker's spec, or anything that says who is bad. They get update tensors and worker ids only. Judges will check this first.

---

## Repo layout

```
antidoteml/
  CLAUDE.md
  README.md
  pyproject.toml          [4]
  antidote/
    types.py              [4] frozen
    ml/                   [1] data.py model.py flat.py train.py evaluate.py
    attacks/              [2] trigger.py poison.py tamper.py
    defense/              [2] features.py detector.py
    system/
      coordinator.py      [3]
      aggregate.py        [3]
      runlog.py           [3]
      run.py              [3]
      config.py           [4]
      worker.py           [4]
      pool.py             [4]
  configs/                [4] smoke.yaml clean_off.yaml clean_on.yaml backdoor_off.yaml
                              backdoor_on.yaml faulty_off.yaml faulty_on.yaml
  experiments/
    plot_run.py           [4]
    notes.md              [2]
  tests/
  runs/                   generated, git-ignored
  docs/AI_USAGE.md        append-only, one section per person
```

---

## The contract

**An update** is a flat 1-D `float32` CPU tensor: `weights_after_training - weights_before`. The model has **no BatchNorm** (it stores extra state outside the parameters, which breaks flattening). Use GroupNorm or nothing.

Functions take plain arguments, not a config object, so no folder has to import another folder's config class.

```python
# antidote/types.py  [4]
@dataclass
class Score:
    worker_id: int
    norm_ratio: float | None     # norm / median norm this round
    cosine: float | None         # cosine similarity to the coordinate-wise median update
    anomaly: float               # >= 0, higher = more suspicious
    reputation: float            # 0..1, starts at 1
    status: str                  # "trusted" | "suspect" | "ejected"
    reason: str                  # "" when trusted
```

```python
# antidote/ml/  [1]
MEAN, STD                                           # normalization constants, attacks imports these
CLASS_NAMES                                         # list of 43 names
load_data(dataset, num_workers, seed) -> (splits, x_test, y_test)   # splits = [(x, y), ...]
make_model(num_classes) -> nn.Module
get_flat(model) -> Tensor
set_flat(model, flat) -> None
local_train(model, x, y, epochs, batch_size, lr, momentum, seed, device) -> None   # trains in place
evaluate(flat, x_test, y_test, target_class, device) -> (clean_acc, asr)           # uses attacks.apply_trigger
```

```python
# antidote/attacks/  [2]      spec = that worker's dict from `attackers:` in the config, or None
apply_trigger(x) -> x                               # [B,3,32,32] in, same shape out, new tensor
maybe_poison(x, y, spec, round, target_class, seed) -> (x, y)   # no-op if spec is None, mode != "scale", or round < start_round
maybe_tamper(delta, spec, round, seed) -> delta                 # modes: scale | noise | nan. Same no-op rule.
```

```python
# antidote/defense/  [2]
compute_features(deltas) -> list[dict]                          # norm_ratio, cosine per row. No state.
Detector(warmup, threshold, strikes, window, hard_norm_ratio)
Detector.inspect(deltas, worker_ids, round) -> list[Score]      # same order as worker_ids
Detector.ejected() -> set[int]
```

```python
# antidote/system/aggregate.py  [3]
aggregate(deltas, method, trim_frac=0.2) -> Tensor[D]           # deltas: Tensor[n, D]. method: mean | median | trimmed_mean
```

```python
# antidote/system/pool.py  [4]      Person 4 builds it, Person 3 only calls it
class WorkerPool:
    def __init__(self, cfg, splits, specs): ...
    def run_round(self, global_flat, round, active_ids) -> dict[int, Tensor | None]
        # worker_id -> update.  None = that worker timed out or crashed this round
    def status(self) -> dict[int, str]
        # last round, per worker: "ok" | "timeout" | "failed".  Used for logging only.
    def close(self): ...

InProcessPool(WorkerPool)      # plain loop, ships first
MultiprocessPool(WorkerPool)   # one process per worker, ships after the loop works end to end
make_pool(cfg, splits, specs) -> WorkerPool      # picks by cfg.execution
```

The coordinator never needs to know which pool is running.

**The worker's whole job** (Person 4 writes it). An honest and a compromised worker run identical code. Only `spec` differs.

```python
set_flat(model, global_flat)
x, y  = maybe_poison(x, y, spec, rnd, target_class, seed)
local_train(model, x, y, ...)
delta = get_flat(model) - global_flat
delta = maybe_tamper(delta, spec, rnd, seed)
```

**One round** (Person 3 writes it):

```
active  = worker ids not in detector.ejected()
results = pool.run_round(global_flat, round, active)   # id -> update, or None
drop the None results (timed out or crashed) and note them for the log
scores  = detector.inspect(deltas, ids, round)         # defense on
keep    = rows whose status is "trusted"
if fewer than half are trusted: keep everyone          # safety net
global += aggregate(keep, method, trim_frac)
clean_acc, asr = evaluate(global, ...)
write one log line
```

With defense off: aggregate with `mean`, no detector, drop any non-finite update so the run does not crash, still log `compute_features` so the plot shows the attacker sitting there unnoticed.

### Config (YAML)

```yaml
name: backdoor_on
seed: 42
dataset: gtsrb              # gtsrb | mnist
num_workers: 10
rounds: 40
train_images: null          # null = use every image. A number truncates each worker split to train_images // num_workers.
local_epochs: 1
batch_size: 64
lr: 0.01
momentum: 0.9
target_class: 5             # confirm at kickoff by viewing images: 5 = speed limit 80, 14 = stop
defense: on                 # off = mean, no detector.  on = aggregator below + detector
aggregator: trimmed_mean
trim_frac: 0.2
detector: {warmup: 3, threshold: 3.5, strikes: 3, window: 5, hard_norm_ratio: 5.0}
execution: inprocess        # inprocess | multiprocess
round_timeout_s: 60         # multiprocess only
attackers:                  # worker id -> spec. Omit the section for a clean run.
  6: {mode: scale, start_round: 10, poison_frac: 0.5, scale: 10}
# 2: {mode: noise, start_round: 15}      # faulty GPU
# faults:                                # optional, demo only, multiprocess only
#   - {worker: 3, round: 8, kind: sleep} # kind: sleep | exit. Makes a worker hang or die on purpose.
```

`faults` is separate from `attackers` on purpose. An attacker sends bad numbers. A fault is a machine that hangs or dies. The pool handles faults, the detector handles attackers.

### Run log (`runs/<name>.jsonl`)

One JSON object per line, flushed every round.

```json
{"type":"header","name":"backdoor_on","config":{...},"class_names":[...]}
{"type":"round","round":12,"clean_acc":0.94,"asr":0.03,
 "workers":[{"id":0,"role":"honest","participated":true,"kept":true,"norm_ratio":0.98,"cosine":0.81,"anomaly":0.4,"reputation":1.0,"status":"trusted","reason":""},
            {"id":6,"role":"backdoor","participated":true,"kept":false,"norm_ratio":9.7,"cosine":-0.12,"anomaly":41.2,"reputation":0.49,"status":"suspect","reason":"update 9.7x larger than the group, pointing away from it (cosine -0.12)"}],
 "events":[{"type":"suspect","worker_id":6,"message":"worker 6 flagged (strike 2 of 3)"}]}
{"type":"summary","final_clean_acc":0.948,"final_asr":0.02,"ejected":[{"worker_id":6,"round":13,"reason":"..."}],"false_positives":0,"seconds":412}
```

`role` is ground truth (`honest`, `backdoor`, `faulty`), written by the coordinator from the config for logging only. Replace NaN with `null`. A worker that was ejected, timed out, or crashed gets `participated: false` and `null` scores. For a timeout or crash, add an event with the value from `pool.status()`.

---

## Person 1: ML

Goal: a model that trains well, can be turned into a vector and back, and can be scored.

1. **`data.py`**: load GTSRB with torchvision, resize to 32x32, scale to 0..1, normalize with `MEAN` and `STD` (use 0.5 for both). **Cache everything as one `.pt` file** the first time, since reading image files every run is the biggest slowdown. Seeded shuffle, split into `num_workers` equal piles. Hardcode the 43 class names. **If GTSRB eats more than 45 minutes, switch to MNIST** (repeat the channel to 3, resize to 32) and move on.
2. **`model.py`**: three conv blocks (32, 64, 128 channels, 3x3, GroupNorm, ReLU, maxpool), flatten, linear 256, linear `num_classes`. About 630k parameters. Test: the model has zero buffers.
3. **`flat.py`**: `get_flat` concatenates every parameter, detached, on CPU. `set_flat` copies slices back. Test: round trip is exact.
4. **`train.py`**: `local_train` uses SGD, a fresh optimizer each call, and batches by slicing a seeded random permutation (much faster than a DataLoader for in-memory tensors).
5. **`evaluate.py`**: clean accuracy on a fixed 3,000-image test subset. Attack success rate: take test images whose true class is not the target, stamp the trigger, report the fraction predicted as the target. Keep one reusable eval model.
6. **Sanity check before anything else:** train one model on all the data, no federation. It must reach 90% or more. If not, nothing downstream works.

**After the sanity check you are the floater.** This is the lightest seat. First help Person 2 land the backdoor, since you know the training loop best and that task is the biggest risk in the project. Then help with results and the write-up.

Explain to a judge: what a gradient update is, why we send updates instead of data, how attack success rate is measured and why clean accuracy alone misses the backdoor.

## Person 2: Security

Goal: a backdoor that reliably lands when there is no defense, and the detector that catches it. **Landing the backdoor is the riskiest task in the project. No landed backdoor means no demo.** This is a defensive research simulation and only ever runs inside our own training loop.

This is the heaviest seat, so the order matters. Steps 1 to 5 need nobody else, which fills the time while everyone builds.

**Attacks (about 1.5 hours, testable on random tensors)**

1. **`trigger.py`**: `apply_trigger` stamps a 4x4 yellow square in the bottom-right corner. Yellow is RGB (1, 1, 0). Convert with `(value - MEAN) / STD`, which gives (1, 1, -1). Return a new tensor. Test: only 16 pixels change.
2. **`poison.py`**: `maybe_poison`, from `start_round`, stamps the trigger on `poison_frac` of the worker's images and relabels them to the target class. Any image can be poisoned. Build the poisoned set once and cache it on first use.
3. **`tamper.py`**: `maybe_tamper` modes are `scale` (multiply by `scale`), `noise` (replace with random values 10 times the original spread), and `nan` (set 1% of entries to NaN).

**Detector on synthetic data (about 3 hours)**

4. **Synthetic test data first** (`tests/` helper): 9 vectors that share a direction plus small noise, and 1 outlier with a different direction and 10 times the norm.
5. **`features.py`**: `norm_ratio` (norm divided by the median norm) and `cosine` (similarity to the coordinate-wise median update, 0 if either norm is zero). Non-finite rows get `None`.
6. **`detector.py`**, each round:
   - NaN or Inf in an update: eject immediately, reason "sent non-finite values."
   - Robust z-score on `log(norm_ratio)` and on `cosine`: `z = (v - median) / (1.4826 * MAD + floor)`. Floor 0.05. Median and MAD are used because one extreme attacker would drag a mean and inflate a standard deviation, hiding itself.
   - `anomaly = max(z_norm, -z_cosine, 0)`.
   - Suspect if `round > warmup` and (`anomaly > threshold` or `norm_ratio > hard_norm_ratio`).
   - Eject on `strikes` flags within the last `window` rounds.
   - `reputation = 0.7 * reputation + 0.3 * (0 if suspect else 1)`.
   - Reason string in plain words with numbers.
7. **Tests**: outlier ejected on its third flagged round; NaN worker ejected the same round; 40 clean rounds eject nobody; scores come back in input order; the `defense/` source contains no `spec`, `role`, or `attackers`.

**At integration**

8. **Make the backdoor land.** Target with defense off: attack success at or above 80%, clean accuracy within 2 points of the clean run. Why scaling works: a plain average of 10 updates gives the attacker one tenth of the say, and multiplying by 10 cancels that. If it does not land, change one thing at a time in this order: start later (round 12 to 15), scale up (12 to 15), poison fraction up (0.7), bigger trigger (6x6). If clean accuracy collapses, the scale is too high. Log each attempt in `experiments/notes.md`. Person 1 helps here.
9. **Tune the detector.** Look at the highest honest anomaly in the clean run and set `threshold` about 1.5 times above it. Change config values, not code.

Explain to a judge: how a backdoor differs from sabotage, why the attacker scales its update and what that costs them, what the two features measure, what is textbook (robust aggregation) and what is ours (scoring, reputation, ejection, explanations, the faulty-hardware use).

## Person 3: Systems, coordinator

Goal: the round loop that ties everything together, and an aggregation step that one liar cannot steer.

1. **`aggregate.py` first, on synthetic data.** `mean`; `median` (`deltas.median(dim=0).values`); `trimmed_mean` (sort each column, drop the top and bottom `floor(trim_frac * n)` rows, average the rest, fall back to median if too few remain). `n` shrinks when workers are ejected or time out, so compute from the actual row count every call. Test with 9 vectors that share a direction and 1 outlier at 10 times the norm: mean is dragged away, the other two stay with the honest direction. You need nobody to start this.
2. **`coordinator.py`**: the round loop above, written against `WorkerPool` only. Build it against Person 4's `InProcessPool` and the stubs. Include the safety net, the non-finite guard with defense off, an `attack_started` event, and per-worker log rows.
3. **`runlog.py`**: header, one line per round, summary. Flush every line. Summary counts false positives (ejected workers whose role is honest).
4. **`run.py`**: `python -m antidote.system.run --config configs/backdoor_on.yaml`. Build the pool with `make_pool`, call `pool.close()` in a `finally` block. Print one progress line per round.
5. **Tests**: the smoke config runs end to end and writes a valid log; `None` results are dropped and logged, not passed to the detector; a NaN update with defense off does not crash the run; an ejected worker stops being asked for updates.

Explain to a judge: why the average fails against one liar (1.0, 1.1, 0.9, 1.0, 50 gives 10.8, trimmed gives 1.03), how that connects to Byzantine fault tolerance, what one round does, what the safety net is for.

## Person 4: Systems, runtime

Goal: everything runs from one command, then runs as real parallel workers.

1. **First 30 minutes, blocks everyone:** repo skeleton, `pyproject.toml` (torch, torchvision, pyyaml, numpy, matplotlib, pytest), `.gitignore`, `types.py`, and **every contract function stubbed** so imports work (stub `aggregate` = mean, stub detector trusts everyone, stub `maybe_poison` and `maybe_tamper` return their input, stub `apply_trigger` returns its input). Add `InProcessPool` as a plain loop, a minimal `config.py`, `configs/smoke.yaml`, and a placeholder `coordinator.py` and `run.py` that Person 3 replaces. One passing test. Push to `main`.
2. **`config.py`**: YAML to a dataclass, validation (attacker ids in range, fault ids in range), `set_seed`, `pick_device` (`mps`, then `cuda`, then `cpu`). `configs/smoke.yaml`: 5 workers, 3 rounds, 1,000 training images.
3. **`worker.py`**: the five-line job. Build each worker's model once and reuse it. Every seed is derived from (config seed, worker id, round), so results do not depend on which pool runs.
4. **`pool.py`, `InProcessPool`**, fully tested, plus `make_pool`.
5. **Scenario configs**: the six files in the layout above. Use different worker ids for the backdoor and faulty scenarios.
6. **`MultiprocessPool`**, once the loop works end to end (`execution: multiprocess`):
   - `spawn` context, one process per worker, one pipe per worker.
   - Workers run on CPU with `torch.set_num_threads(1)`, so ten processes do not fight over cores or one GPU.
   - Each process receives its data split and spec once at startup and sends a ready message. The pool waits for every ready message before round 1.
   - Each round: send (round, global weights) to the active workers, collect updates against one shared deadline.
   - Every message carries its round number. A reply from an old round is discarded.
   - Tensors cross the pipe as numpy arrays.
   - A worker that misses the deadline returns `None` for that round. A dead process or a broken pipe marks the worker failed for the rest of the run. No restarts.
   - `close()` sends shutdown, joins with a timeout, terminates stragglers.
   - Same interface, so nobody else changes anything. This is the systems work you show judges.
7. **Pool tests**: both pools give identical updates for the same seed on the smoke config; a `sleep` fault returns `None` without holding the round past the deadline; an `exit` fault marks the worker failed and the run continues; a late reply from a previous round is discarded.
8. **`experiments/plot_run.py`**: from one or two run logs, save a PNG. Top panel: clean accuracy and attack success per round with lines at attack start and ejection. Bottom panel: `norm_ratio` per worker with the attacker highlighted. This is the demo visual.

Explain to a judge: how workers and the coordinator talk, what happens when a worker hangs or crashes, why a late reply is thrown away, why both pools give the same numbers.

---

## Timeline

| When | What |
|---|---|
| 0:00 to 0:30 | Contract call. Person 4 pushes the skeleton with stubs and `InProcessPool`. |
| 0:30 to 3:00 | Everyone builds in parallel. Security and aggregation test on fake tensors. |
| 3:00 | A clean run reaches 90% through the real coordinator. |
| 4:00 | The backdoor lands with defense off. Persons 1 and 2 work on this together. **If not, everyone helps.** |
| 5:00 | Defense on: attack success near 0, attacker ejected, no honest worker ejected. |
| 5:00 on | Multiprocess pool, faulty-GPU runs, plots, README, video, Devpost. |

**Done means:** clean run at or above 90%. `backdoor_off` attack success at or above 80%. `backdoor_on` attack success at or below 10% with the attacker ejected within 5 rounds and zero false positives. `faulty_on` ejects the faulty worker. `clean_on` ejects nobody.

**Cut until the 5:00 milestone is green:** Krum, uneven data splits, two attackers, stealth attack, API, frontend.

---

## Git and Claude Code

- Branch per person: `ml`, `security`, `coordinator`, `runtime`. Small pull requests to `main`. Person 4 merges and only checks that you touched your own paths and tests pass.
- `pytest -q` must pass before every pull request.
- Never commit datasets, `.pt` files, or `runs/`.
- The repo-root `CLAUDE.md` is loaded by Claude Code automatically and carries the ownership rules.
- Log what you used AI for in your section of `docs/AI_USAGE.md` as you go. The hackathon requires the disclosure and judges can ask anyone to explain any part.

**Kickoff prompt (fill in your number and paths):**

```
I am Person <N> on AntidoteML. Read ANTIDOTEML_PLAN.md. I own <paths> and their
tests only. Do not edit anything else: if another file needs a change, write
me a note for its owner instead. Follow the contract signatures exactly.
Write tests first, then the code. Start with step 1 of my section.
```

---

## If something breaks

| Symptom | Fix |
|---|---|
| Clean accuracy stuck near 2% | Train one model on all data alone. Check labels still line up with images after the split. |
| Attack success stays near 0 with defense off | Person 2, step 8 list, in order. |
| Clean accuracy collapses when the attack starts | Scale too high. Lower it. |
| Defense on, still backdoored | Plot anomaly per worker. Lower `threshold` or `strikes`. |
| Honest workers ejected | Raise `warmup` to 5, raise `threshold`, check the MAD floor. |
| Loss goes NaN in the faulty run with defense off | Non-finite guard missing in the coordinator. |
| A run takes 30 minutes | Dataset not cached, or a DataLoader on in-memory data. |
| Same config gives different results | Unseeded randomness somewhere. |
| Multiprocess is slower than in-process | Workers are not pinned to one thread, or they are sharing a GPU. |
| Every worker times out in round 1 | Process startup counted against the deadline. The ready handshake is missing. |
