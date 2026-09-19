"""PLACEHOLDER for Person 3. Enough of a round loop to prove the wiring works.

Person 3 replaces this whole file with the real loop: detector, safety net,
non-finite guard, events and per-worker log rows.
"""

import torch

from antidote.defense import Detector
from antidote.ml import evaluate, get_flat, make_model
from antidote.ml.data import CLASS_NAMES
from antidote.system.aggregate import aggregate
from antidote.system.runlog import RunLog


def run_training(cfg, pool, splits, x_test, y_test, log_path, device="cpu"):
    """Run every round and write the log. Returns the final global weights."""
    log = RunLog(log_path)
    detector = Detector(**cfg.detector)
    global_flat = get_flat(make_model(len(CLASS_NAMES)))

    log.write({"type": "header", "name": cfg.name, "config": vars(cfg), "class_names": CLASS_NAMES})
    try:
        for rnd in range(1, cfg.rounds + 1):
            active = [w for w in range(cfg.num_workers) if w not in detector.ejected()]
            results = pool.run_round(global_flat, rnd, active)

            ids = [w for w in active if results.get(w) is not None]
            deltas = torch.stack([results[w] for w in ids]) if ids else None
            if deltas is not None:
                detector.inspect(deltas, ids, rnd)
                global_flat = global_flat + aggregate(deltas, cfg.aggregator, cfg.trim_frac)

            clean_acc, asr = evaluate(global_flat, x_test, y_test, cfg.target_class, device)
            log.write(
                {
                    "type": "round",
                    "round": rnd,
                    "clean_acc": clean_acc,
                    "asr": asr,
                    "workers": [{"id": w, "participated": results.get(w) is not None} for w in active],
                    "events": [],
                }
            )
            print(f"round {rnd}/{cfg.rounds} clean_acc {clean_acc:.3f} asr {asr:.3f}")

        log.write({"type": "summary", "final_clean_acc": clean_acc, "final_asr": asr, "ejected": []})
    finally:
        log.close()
    return global_flat
