# AntidoteML — Team Plan

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
| 1 | ML | `antidote/ml/` | data, model, flat vectors, local training, evaluation, sanity script. Floater once done. |
| 2 | Security | `antidote/attacks/`, `antidote/defense/` | trigger, poisoning, tampering, landing the backdoor, update features, detector, ejection |
| 3 | Systems: coordinator | `antidote/system/coordinator.py`, `aggregate.py`, `runlog.py`, `run.py` | round loop, robust aggregators, safety net, run log, CLI |
| 4 | Systems: runtime | `antidote/system/config.py`, `worker.py`, `pool.py`, `antidote/types.py`, repo skeleton, `configs/`, `experiments/` | skeleton and stubs, config, the worker job, the worker pool (in-process, then real parallel processes), scenario configs, plot script |

Robust aggregation sits with Systems on purpose: surviving a node that lies is a classic distributed systems problem (Byzantine fault tolerance).

Tests live in `tests/` and are named after your area: `test_ml_*.py`, `test_attacks_*.py`, `test_defense_*.py`, `test_coordinator_*.py`, `test_runtime_*.py`.

### The three rules

1. **Edit only what you own.** Need a change elsewhere? Message the owner with the file, the problem, and the fix you want.
2. **The contract below is frozen** after the kickoff call. Changing it needs all four to agree. Person 4 makes the edit.
3. **The defense never sees the answer.** Nothing in `defense/` may read the `attackers:` config, a worker's spec, or anything that says who is bad. It gets update tensors and worker ids only. Judges will check this first.

---

## Repo layout

```
antidoteml/
  CLAUDE.md
  ANTIDOTEML_PLAN.md
  README.md
  pyproject.toml          [4]
  antidote/
    types.py              [4] frozen
    ml/                   [1] data.py model.py flat.py train.py evaluate.py sanity.py
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
  configs/                [4] smoke.yaml clean_off/on.yaml backdoor_off/on.yaml faulty_off/on.yaml
  experiments/            [4] plot_run.py      (notes.md is append-only, one section per person)
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
# antidote/defense/  [2]      never sees specs, roles, or the attackers config
compute_features(deltas) -> list[dict]                          # norm_ratio, cosine per row. No state.
Detector(warmup, threshold, strikes, window, hard_norm_ratio)
Detector.inspect(deltas, worker_ids, round) -> list[Score]      # deltas: Tensor[n, D]. Same order as worker_ids.
Detector.ejected() -> set[int]
```

```python
# antidote/system/aggregate.py  [3]
aggregate(deltas, method, trim_frac=0.2) -> Tensor[D]           # deltas: Tensor[n, D]. method: mean | median | trimmed_mean

# antidote/system/pool.py  [4]      the one interface between the two systems people
class WorkerPool:
    def __init__(self, cfg, splits, specs): ...                 # specs: {worker_id: attack dict}, absent = honest
    def run_round(self, global_flat, round, active_ids) -> dict[int, Tensor | None]
        # worker_id -> update. None = that worker timed out or crashed this round.
    def close(self) -> None: ...
make_pool(cfg, splits, specs) -> WorkerPool                     # picks InProcessPool or MultiprocessPool from cfg.execution
```

**The worker's whole job** (Person 4 writes it, in `worker.py`). An honest and a compromised worker run identical code. Only `spec` differs.

```python
set_flat(model, global_flat)
x, y  = maybe_poison(x, y, spec, rnd, target_class, seed)
local_train(model, x, y, ...)
delta = get_flat(model) - global_flat
delta = maybe_tamper(delta, spec, rnd, seed)
```

**One round** (Person 3 writes it, in `coordinator.py`):

```
active  = worker ids not in detector.ejected()
results = pool.run_round(global_flat, round, active)   # drop None results, log them as events
deltas  = the updates that came back
scores  = detector.inspect(deltas, ids, round)        # defense on
keep    = rows whose status is "trusted"
if fewer than half are trusted: keep everyone         # safety net
global += aggregate(keep, method)
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
attackers:                  # worker id -> spec. Omit the section for a clean run.
  6: {mode: scale, start_round: 10, poison_frac: 0.5, scale: 10}
# 2: {mode: noise, start_round: 15}      # faulty GPU
```

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

`role` is ground truth (`honest`, `backdoor`, `faulty`), written by the coordinator from the config for logging only. Replace NaN with `null`.

---

## Person 1 — ML

Goal: a model that trains well, can be turned into a vector and back, and can be scored. Then float to wherever the risk is.

1. **`model.py`** — three conv blocks (32, 64, 128 channels, 3x3, GroupNorm(8), ReLU, maxpool), flatten to 2048, linear 256, linear `num_classes`. 629,291 parameters for 43 classes. Test: zero buffers, exact parameter count.
2. **`flat.py`** — `get_flat` concatenates every parameter, detached, on CPU, float32. `set_flat` copies slices back under `no_grad` and raises `ValueError` on a length mismatch. Test: round trip is exact.
3. **`train.py`** — `local_train`: fresh SGD optimizer each call, cross-entropy, batch by slicing a seeded `randperm` (much faster than a DataLoader for in-memory tensors). Callers pass a different seed per worker and round: `seed + 1000 * worker_id + round`.
4. **`data.py`** — GTSRB through torchvision, 32x32, normalized with `MEAN = STD = 0.5`. **Cache everything as one uint8 `.pt` file** the first time. `split_even` does a seeded shuffle into near-equal piles. Hardcode the 43 class names. Optional `max_train` and `max_test` for smoke runs. MNIST fallback (repeat the channel to 3). **If GTSRB eats more than 45 minutes, switch to MNIST and move on.**
5. **`evaluate.py`** — one reusable eval model. Infer the class count from the vector length (the last layer adds 257 values per class). Clean accuracy on a fixed 3,000-image subset (`max_eval=None` for the full set). Attack success rate: non-target test images, trigger stamped, fraction predicted as the target. Import `apply_trigger` lazily with a do-nothing fallback so this works before Person 2 merges.
6. **`sanity.py`** — `python -m antidote.ml.sanity`. Part 1: one model on all data, must pass 90%. Part 2: 10 workers with plain averaging for 10 rounds, accuracy must climb. **If Part 1 fails, nothing downstream works. Tell the team immediately.**
7. **Hand-offs** — tell Person 2 that `MEAN` and `STD` are importable. Tell Persons 3 and 4 the exact call signatures and the seed rule. Report seconds per worker per round so Person 4 can set the multiprocess timeout.
8. **Floater, once the sanity check passes** — (a) pair with Person 2 to land the backdoor, since you know the training loop best and it is the project's biggest risk, (b) speed: keep data on the device, reuse models, (c) final full-test-set numbers, README quick start, and the results section of the write-up.

Explain to a judge: what an update is and why we send updates instead of data, why there is no BatchNorm, how attack success rate is measured and why clean accuracy alone misses the backdoor.

## Person 2 — Security (attacks and defense)

Goal: a backdoor that reliably lands with no defense, and a detector that catches it. **This is the heaviest seat. Follow this order so you are never waiting on anyone.** This is a defensive research simulation and only ever runs inside our own training loop.

**Hard rule:** nothing in `defense/` may import from `attacks/`, read a worker's spec, the `attackers:` config, or a role. It gets update tensors and worker ids only. Judges will check this first.

1. **Attack functions (about 1.5 h, random tensors only)**
   - `trigger.py` — `apply_trigger`: 4x4 yellow square, bottom-right. Yellow is RGB (1, 1, 0), converted with `(value - MEAN) / STD` to (1, 1, -1). Return a new tensor. Test: only 16 pixels per image change.
   - `poison.py` — `maybe_poison`: from `start_round`, stamp the trigger on `poison_frac` of the worker's images and relabel them to the target class. Any image can be poisoned. Build the poisoned set once and cache it.
   - `tamper.py` — `maybe_tamper` modes: `scale` (multiply), `noise` (random values, 10 times the original spread), `nan` (1% of entries).
2. **Detector on synthetic data (about 3 h, needs nobody)**
   - Test helper: 9 vectors sharing a direction plus small noise, 1 outlier with a different direction and 10 times the norm.
   - `features.py` — `norm_ratio` (norm over median norm) and `cosine` (similarity to the coordinate-wise median update, 0 if a norm is zero). Non-finite rows get `None`.
   - `detector.py`, each round: NaN or Inf means eject immediately. Robust z-score on `log(norm_ratio)` and `cosine`: `z = (v - median) / (1.4826 * MAD + 0.05)`. `anomaly = max(z_norm, -z_cosine, 0)`. Suspect if past `warmup` and (`anomaly > threshold` or `norm_ratio > hard_norm_ratio`). Eject on `strikes` flags within `window` rounds. `reputation = 0.7 * rep + 0.3 * (0 if suspect else 1)`. Reason string in plain words with numbers.
   - Tests: outlier ejected on its third flagged round, NaN worker ejected the same round, 40 clean rounds eject nobody, scores in input order, no `spec`, `role`, or `attackers` anywhere in `defense/`.
3. **Land the backdoor (at integration, with Person 1).** Target with defense off: attack success at or above 80%, clean accuracy within 2 points of the clean run. Scaling works because a plain average of 10 updates gives the attacker one tenth of the say, and multiplying by 10 cancels that. If it does not land, change one thing at a time: start later (round 12 to 15), scale up (12 to 15), poison fraction up (0.7), bigger trigger (6x6). If clean accuracy collapses, lower the scale. Log each attempt in `experiments/notes.md`.
4. **Tune the detector on real runs.** Find the highest honest anomaly in the clean run, set `threshold` about 1.5 times above it. Change config values, not code.

Explain to a judge: how a backdoor differs from sabotage, why the attacker scales and what that costs them, why median and MAD instead of mean and standard deviation, what is textbook and what is ours (scoring, reputation, ejection, explanations, the faulty-hardware use).

## Person 3 — Systems: coordinator

Goal: the brain of the training run. It trusts nothing a worker sends until the defense has looked at it.

1. **`aggregate.py`** — stack updates to `n x D`. `mean`; `median` (`deltas.median(dim=0).values`); `trimmed_mean` (sort each column, drop the top and bottom `floor(trim_frac * n)` rows, average the rest, fall back to median if too few remain). `n` shrinks as workers are ejected, so compute from the actual row count. Test with Person 2's style of synthetic data: mean is dragged by the outlier, the other two are not.
2. **`coordinator.py`** — the round loop from the contract: active ids, `pool.run_round`, drop `None` results and log a `worker_timeout` or `worker_failed` event, `detector.inspect`, keep trusted rows, safety net (fewer than half trusted means keep everyone), `aggregate`, update the global vector, `evaluate`, log. With defense off: `mean`, no detector, drop non-finite updates so the run survives, still log `compute_features`. Emit `attack_started`, `suspect`, `ejected` events.
3. **`runlog.py`** — header, one JSON line per round, summary. Flush every line. NaN becomes `null`. Per-worker rows include `role` from the config (logging only), and ejected workers get `participated: false`. Summary has full-test-set accuracy (`max_eval=None`), ejections, false positives (ejected workers whose role is honest), seconds.
4. **`run.py`** — `python -m antidote.system.run --config configs/backdoor_on.yaml`. Loads config, seeds, builds data, pool, detector, runs, prints one line per round.
5. Until Person 4's pool exists, test the loop with a fake pool that returns random vectors.

Explain to a judge: what one round does, why the coordinator must tolerate a lying worker (Byzantine fault tolerance), why the median survives one liar and the mean does not.

## Person 4 — Systems: runtime

Goal: make it run, then make it run as real parallel workers.

1. **First 30 minutes, blocks everyone:** repo skeleton, `pyproject.toml` (torch, torchvision, pyyaml, numpy, matplotlib, pytest), `.gitignore` (`.venv/`, `data/`, `runs/`, `*.pt`, `__pycache__/`), `types.py`, and **every contract function stubbed** so imports work. One passing test. Push to `main`.
2. **`config.py`** — YAML to a dataclass, validation (attacker ids in range), `set_seed`, `pick_device` (`mps`, `cuda`, `cpu`).
3. **`worker.py`** — the five-line job. Build each worker's model once and reuse it. Seed is `seed + 1000 * worker_id + round`.
4. **`pool.py`, step one: `InProcessPool`** — a plain loop over workers. About 30 minutes. This unblocks Person 3 and the first end-to-end run.
5. **Configs** — `smoke.yaml` (5 workers, 3 rounds, 1,000 images), plus `clean`, `backdoor`, `faulty`, each `_off` and `_on`. Person 2 tells you the attack values.
6. **`pool.py`, step two: `MultiprocessPool`** — one process per worker with `torch.multiprocessing` (spawn), each holding its own data and model. Broadcast global weights through pipes or queues, collect updates in parallel. Per-round timeout returns `None` for a slow worker. A crashed process returns `None` and the run continues. Keep each process on CPU unless sharing one GPU proves stable. Same interface, so nobody else changes anything.
7. **`experiments/plot_run.py`** — from one or two run logs, save a PNG: clean accuracy and attack success per round with lines at attack start and ejection, and `norm_ratio` per worker with the attacker highlighted. This is the demo visual.

Explain to a judge: how the coordinator and workers talk, what happens when a worker hangs or crashes, why in-process and multiprocess give the same results.

---

## Timeline

| When | What |
|---|---|
| 0:00–0:30 | Everyone agrees the contract on a call. Person 4 pushes the skeleton with stubs. |
| 0:30–3:00 | All four build in parallel. Person 2 tests on fake tensors. Person 3 tests against a fake pool. Person 4 ships `InProcessPool` early. |
| 3:00 | Integrate. Clean run reaches 90%. |
| 4:00 | Backdoor lands with defense off. Persons 1 and 2 pair on this. **If it still fails, everyone helps.** |
| 5:00 | Defense on: attack success near 0, attacker ejected, no honest worker ejected. |
| 5:00 on | Multiprocess workers, faulty-GPU runs, plots, README, video, Devpost. |

**Done means:** clean run at or above 90%. `backdoor_off` attack success at or above 80%. `backdoor_on` attack success at or below 10% with the attacker ejected within 5 rounds and zero false positives. `faulty_on` ejects the faulty worker. `clean_on` ejects nobody.

**Cut until the 5:00 milestone is green:** Krum, uneven data splits, two attackers, stealth attack, API, frontend.

---

## Git and Claude Code

- Branch per person: `ml`, `security`, `coordinator`, `runtime`. Small pull requests to `main`. Person 4 merges and only checks that you touched your own paths and tests pass.
- `pytest -q` must pass before every pull request.
- Never commit datasets, `.pt` files, or `runs/`.
- The repo-root `CLAUDE.md` is loaded by Claude Code automatically and carries the ownership rules.
- Log what you used AI for in your section of `docs/AI_USAGE.md` as you go. The hackathon requires the disclosure and judges can ask anyone to explain any part.

**Kickoff prompt (fill in your number and folder):**

```
I am Person <N> on AntidoteML. Read ANTIDOTEML_PLAN.md. I own <folder> and its
tests only. Do not edit anything else: if another folder needs a change, write
me a note for its owner instead. Follow the contract signatures exactly.
Write tests first, then the code. Start with step 1 of my section.
```

---

## If something breaks

| Symptom | Fix |
|---|---|
| Clean accuracy stuck near 2% | Train one model on all data alone. Check labels still line up with images after the split. |
| Attack success stays near 0 with defense off | Person 2, step 3 list, in order. |
| Clean accuracy collapses when the attack starts | Scale too high. Lower it. |
| Defense on, still backdoored | Plot anomaly per worker. Lower `threshold` or `strikes`. |
| Honest workers ejected | Raise `warmup` to 5, raise `threshold`, check the MAD floor. |
| Loss goes NaN in the faulty run with defense off | Non-finite guard missing in the coordinator. |
| A run takes 30 minutes | Dataset not cached, or a DataLoader on in-memory data. |
| Same config gives different results | Unseeded randomness somewhere. |
