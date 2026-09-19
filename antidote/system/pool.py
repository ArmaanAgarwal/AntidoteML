"""Worker pools. Owner: Person 4. Person 3 only calls this.

The coordinator talks to workers through WorkerPool and never learns which
implementation is running. MultiprocessPool lands in step 6.
"""

from antidote.ml import make_model
from antidote.ml.data import CLASS_NAMES
from antidote.system.worker import run_worker_round


class WorkerPool:
    """Interface. See ANTIDOTEML_PLAN.md for the contract."""

    def __init__(self, cfg, splits, specs):
        self.cfg = cfg
        self.splits = splits
        self.specs = specs

    def run_round(self, global_flat, round, active_ids):
        """worker_id -> update. None means that worker timed out or crashed."""
        raise NotImplementedError

    def status(self):
        """Last round, per worker: "ok" | "timeout" | "failed". Logging only."""
        raise NotImplementedError

    def close(self):
        pass


class InProcessPool(WorkerPool):
    """A plain loop. One model per worker, built once and reused."""

    def __init__(self, cfg, splits, specs):
        super().__init__(cfg, splits, specs)
        self.device = "cpu"
        self._models = {}
        self._status = {}

    def _model(self, worker_id):
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
        raise NotImplementedError("MultiprocessPool lands in Person 4 step 6")
    raise ValueError(f"unknown execution: {cfg.execution}")
