# AI usage

Append-only. Add one line to your own section after each task. Do not edit
another person's section.

## Person 1: ML

## Person 2: Security

- Cursor helped write and test the trigger, poisoning, and update-tampering modules; I reviewed the deterministic seeding, caching, and tensor behavior.
- Cursor helped create and validate the six clean, backdoor, and faulty-worker scenario configs; I reviewed every scenario value.
- Cursor helped implement and test the one/two-run JSONL plotting tool; I reviewed its metric, attacker, attack-start, and ejection rendering.

## Person 3: Systems, coordinator

- Codex implemented robust aggregation, coordinator round handling, JSONL logging, CLI wiring, and Person 3 tests.
- Codex added coordinator compatibility for optional fault metadata and worker pools without status reporting.

## Person 4: Systems, runtime

- Claude Code wrote the repo skeleton, stubs, `config.py`, `worker.py`, `InProcessPool`, `configs/smoke.yaml` and the first tests from step 1 of the plan. I reviewed every file.
- Step 2: Claude Code wrote `tests/test_config_validate.py` and the full `config.py` validation (id ranges, known execution, aggregator, dataset, attack mode and fault kind values) from my description. I chose what to validate and checked each message.
- Step 3 and 4: Claude Code wrote `tests/test_worker_round.py` and `tests/test_pool_inprocess.py` first, then the worker update coercion and the `truncate_splits` cap for `train_images` in `InProcessPool`. I picked the seed derivation and reviewed the tests.
- Step 5: Claude Code wrote `tests/test_config_scenarios.py` and generated the six scenario configs from it. I chose the worker ids, the attack start rounds and the off and on pairing rule.
- Step 6: Claude Code wrote `MultiprocessPool` (spawn, one pipe per worker, ready handshake, one shared round deadline, stale reply discard, faults inside the worker, close with terminate) from my design notes. I walked through the message flow and the failure paths line by line.
- Step 7: Claude Code wrote the pool tests (sleep fault, exit fault, stale reply, close leaves no children, both pools agree). The tests found a real deadlock where the pool blocked writing an update into a full pipe, and I had it fix the pool so work only goes to a worker that has come back from its last job.
