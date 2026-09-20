"""The distributed training round loop. Owner: Person 3."""

import time

import torch

from antidote.defense import Detector, compute_features
from antidote.ml import evaluate, get_flat, make_model
from antidote.ml.data import CLASS_NAMES
from antidote.system.aggregate import aggregate
from antidote.system.runlog import RunLog


def _role(cfg, worker_id):
    """Ground truth role for logs only. Never pass this to the defense."""
    spec = cfg.attackers.get(worker_id)
    if spec is not None:
        return "backdoor" if spec.get("mode") == "scale" else "faulty"
    if any(
        fault.get("worker") == worker_id
        for fault in getattr(cfg, "faults", ())
    ):
        return "faulty"
    return "honest"


def _inactive_row(cfg, worker_id, status="trusted", reason=""):
    return {
        "id": worker_id,
        "role": _role(cfg, worker_id),
        "participated": False,
        "kept": False,
        "norm_ratio": None,
        "cosine": None,
        "anomaly": None,
        "reputation": None,
        "status": status,
        "reason": reason,
    }


def _score_row(cfg, score, kept):
    return {
        "id": score.worker_id,
        "role": _role(cfg, score.worker_id),
        "participated": True,
        "kept": kept,
        "norm_ratio": score.norm_ratio,
        "cosine": score.cosine,
        "anomaly": score.anomaly,
        "reputation": score.reputation,
        "status": score.status,
        "reason": score.reason,
    }


def _feature_row(cfg, worker_id, feature, finite, kept):
    return {
        "id": worker_id,
        "role": _role(cfg, worker_id),
        "participated": True,
        "kept": kept,
        "norm_ratio": feature.get("norm_ratio"),
        "cosine": feature.get("cosine"),
        "anomaly": 0.0 if finite else None,
        "reputation": 1.0 if finite else None,
        "status": "trusted" if finite else "suspect",
        "reason": "" if finite else "non-finite update dropped",
    }


def _attack_events(cfg, rnd):
    events = []
    for worker_id, spec in sorted(cfg.attackers.items()):
        if rnd == int(spec.get("start_round", 0)):
            events.append(
                {
                    "type": "attack_started",
                    "worker_id": worker_id,
                    "message": f"worker {worker_id} attack started",
                }
            )
    return events


def _check_scores(scores, worker_ids):
    if len(scores) != len(worker_ids):
        raise ValueError("detector returned the wrong number of scores")
    if [score.worker_id for score in scores] != list(worker_ids):
        raise ValueError("detector scores are not in worker id order")


def _pool_status(pool):
    status = getattr(pool, "status", None)
    if status is None:
        return {}
    try:
        return status()
    except NotImplementedError:
        return {}


def run_training(cfg, pool, splits, x_test, y_test, log_path, device="cpu"):
    """Run all rounds, write the JSONL log, and return final flat weights."""
    del splits
    started = time.perf_counter()
    log = RunLog(log_path)
    detector = Detector(**cfg.detector) if cfg.defense else None
    global_flat = get_flat(make_model(len(CLASS_NAMES)))
    ejected_details = {}
    last_clean_acc = 0.0
    last_asr = 0.0

    try:
        log.write(
            {
                "type": "header",
                "name": cfg.name,
                "config": vars(cfg),
                "class_names": CLASS_NAMES,
            }
        )

        for rnd in range(1, cfg.rounds + 1):
            already_ejected = detector.ejected() if detector is not None else set()
            active_ids = [
                worker_id
                for worker_id in range(cfg.num_workers)
                if worker_id not in already_ejected
            ]
            results = pool.run_round(global_flat, rnd, active_ids)
            pool_status = _pool_status(pool)
            events = _attack_events(cfg, rnd)

            participating_ids = [
                worker_id
                for worker_id in active_ids
                if results.get(worker_id) is not None
            ]
            updates = [results[worker_id] for worker_id in participating_ids]
            deltas = torch.stack(updates) if updates else None
            finite_ids = {
                worker_id
                for worker_id, update in zip(participating_ids, updates)
                if torch.isfinite(update).all()
            }

            scores_by_id = {}
            features_by_id = {}
            kept_ids = set()

            if deltas is not None and detector is not None:
                scores = detector.inspect(deltas, participating_ids, rnd)
                _check_scores(scores, participating_ids)
                scores_by_id = {score.worker_id: score for score in scores}
                kept_ids = {
                    score.worker_id
                    for score in scores
                    if score.status == "trusted" and score.worker_id in finite_ids
                }

                if len(kept_ids) < len(participating_ids) / 2:
                    kept_ids = set(finite_ids)
                    events.append(
                        {
                            "type": "safety_net",
                            "message": (
                                "fewer than half of participating workers were "
                                "trusted; kept every finite update"
                            ),
                        }
                    )

                for score in scores:
                    if score.status == "suspect":
                        events.append(
                            {
                                "type": "suspect",
                                "worker_id": score.worker_id,
                                "message": score.reason
                                or f"worker {score.worker_id} flagged",
                            }
                        )
                    elif score.status == "ejected":
                        events.append(
                            {
                                "type": "ejected",
                                "worker_id": score.worker_id,
                                "message": score.reason
                                or f"worker {score.worker_id} ejected",
                            }
                        )
                        ejected_details.setdefault(
                            score.worker_id,
                            {
                                "worker_id": score.worker_id,
                                "round": rnd,
                                "reason": score.reason,
                            },
                        )

            elif deltas is not None:
                features = compute_features(deltas)
                if len(features) != len(participating_ids):
                    raise ValueError("compute_features returned the wrong row count")
                features_by_id = dict(zip(participating_ids, features))
                kept_ids = set(finite_ids)

            for worker_id in active_ids:
                if results.get(worker_id) is None:
                    status = pool_status.get(worker_id, "failed")
                    if status not in ("timeout", "failed"):
                        status = "failed"
                    events.append(
                        {
                            "type": status,
                            "worker_id": worker_id,
                            "message": f"worker {worker_id} {status}",
                        }
                    )

            if kept_ids:
                kept = torch.stack(
                    [results[worker_id] for worker_id in sorted(kept_ids)]
                )
                method = cfg.aggregator if cfg.defense else "mean"
                global_flat = global_flat + aggregate(kept, method, cfg.trim_frac)

            last_clean_acc, last_asr = evaluate(
                global_flat, x_test, y_test, cfg.target_class, device
            )

            worker_rows = []
            for worker_id in range(cfg.num_workers):
                if worker_id not in active_ids:
                    detail = ejected_details.get(worker_id, {})
                    worker_rows.append(
                        _inactive_row(
                            cfg,
                            worker_id,
                            status="ejected",
                            reason=detail.get("reason", "worker ejected"),
                        )
                    )
                elif results.get(worker_id) is None:
                    worker_rows.append(_inactive_row(cfg, worker_id))
                elif detector is not None:
                    worker_rows.append(
                        _score_row(
                            cfg,
                            scores_by_id[worker_id],
                            worker_id in kept_ids,
                        )
                    )
                else:
                    worker_rows.append(
                        _feature_row(
                            cfg,
                            worker_id,
                            features_by_id[worker_id],
                            worker_id in finite_ids,
                            worker_id in kept_ids,
                        )
                    )

            log.write(
                {
                    "type": "round",
                    "round": rnd,
                    "clean_acc": last_clean_acc,
                    "asr": last_asr,
                    "workers": worker_rows,
                    "events": events,
                }
            )
            print(
                f"round {rnd}/{cfg.rounds} "
                f"clean_acc {last_clean_acc:.3f} asr {last_asr:.3f}"
            )

        ejected = [ejected_details[key] for key in sorted(ejected_details)]
        false_positives = sum(
            _role(cfg, detail["worker_id"]) == "honest" for detail in ejected
        )
        log.write(
            {
                "type": "summary",
                "final_clean_acc": last_clean_acc,
                "final_asr": last_asr,
                "ejected": ejected,
                "false_positives": false_positives,
                "seconds": time.perf_counter() - started,
            }
        )
    finally:
        log.close()

    return global_flat
