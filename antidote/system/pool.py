"""Worker pools. Owner: Person 4. Person 3 only calls this.

The coordinator talks to workers through WorkerPool and never learns which
implementation is running. MultiprocessPool lands in step 4.
"""

from antidote.ml import make_model
from antidote.ml.data import CLASS_NAMES
from antidote.system.worker import run_worker_round


def truncate_splits(splits, train_images, num_workers):
    """Cap each worker split so a whole run uses about train_images images.

    load_data's signature is frozen, so the cap cannot live there. It lives
    here instead, applied once when the pool is built, which keeps the smoke
    config down to a few seconds without touching Person 1's loader.
    """
    if not train_images:
        return list(splits)
    per_worker = max(1, train_images // num_workers)
    return [(x[:per_worker], y[:per_worker]) for x, y in splits]


class WorkerPool:
    """Interface. See ANTIDOTEML_PLAN.md for the contract."""

    def __init__(self, cfg, splits, specs):
        self.cfg = cfg
        self.splits = truncate_splits(splits, cfg.train_images, cfg.num_workers)
        self.specs = dict(specs or {})

    def run_round(self, global_flat, round, active_ids):
        """worker_id -> update. None means that worker timed out or crashed."""
        raise NotImplementedError

    def status(self):
        """Last round, per worker: "ok" | "timeout" | "failed". Logging only."""
        raise NotImplementedError

    def close(self):
        pass


class InProcessPool(WorkerPool):
    """A plain loop. One model per worker, built once and reused.

    Workers stay on the cpu even when a gpu is free, so this pool and the
    multiprocess pool produce the same numbers for the same seed.
    """

    def __init__(self, cfg, splits, specs):
        super().__init__(cfg, splits, specs)
        self.device = "cpu"
        self._models = {}
        self._status = {}

    def _model(self, worker_id):
        """One model per worker, kept between rounds. Building a fresh model
        every round would be slower and would waste a draw of random weights,
        which set_flat overwrites on the next line anyway."""
        if worker_id not in self._models:
            self._models[worker_id] = make_model(len(CLASS_NAMES))
        return self._models[worker_id]

    def run_round(self, global_flat, round, active_ids):
        results = {}
        self._status = {}
        for wid in active_ids:
            x, y = self.splits[wid]
            try:
                results[wid] = run_worker_round(
                    self._model(wid),
                    global_flat,
                    x,
                    y,
                    self.specs.get(wid),
                    wid,
                    round,
                    self.cfg,
                    self.device,
                )
                self._status[wid] = "ok"
            except Exception:
                # Nothing to restart in this pool, so the worker simply has no
                # update this round. The coordinator drops it and logs it.
                results[wid] = None
                self._status[wid] = "failed"
        return results

    def status(self):
        return dict(self._status)

    def close(self):
        self._models.clear()


def make_pool(cfg, splits, specs):
    """Pick a pool by cfg.execution."""
    if cfg.execution == "inprocess":
        return InProcessPool(cfg, splits, specs)
    if cfg.execution == "multiprocess":
        raise NotImplementedError("MultiprocessPool lands in Person 4 step 4")
    raise ValueError(f"unknown execution: {cfg.execution}")
