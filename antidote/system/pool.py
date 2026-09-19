"""Worker pools. Owner: Person 4. Person 3 only calls this.

The coordinator talks to workers through WorkerPool and never learns which
implementation is running. Both pools keep workers on the cpu and take every
seed from (config seed, worker id, round), so the same config gives the same
numbers whichever pool runs it.
"""

import multiprocessing as mp
import os
import time
from dataclasses import replace
from multiprocessing.connection import wait

import numpy as np
import torch

from antidote.ml import make_model
from antidote.ml.data import CLASS_NAMES
from antidote.system.worker import run_worker_round

# Message kinds on the pipe. Every task and every reply carries its round.
READY = "ready"
TASK = "task"
UPDATE = "update"
SHUTDOWN = "shutdown"

# Starting a process means a fresh python and a fresh torch import, which is
# slow and has nothing to do with how long a round takes. It gets its own
# generous budget so a slow start never looks like a timed out round.
STARTUP_TIMEOUT_S = 120.0
CLOSE_TIMEOUT_S = 5.0


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


def to_numpy(tensor):
    """A tensor on its way into a pipe."""
    return np.ascontiguousarray(tensor.detach().to("cpu", torch.float32).reshape(-1).numpy())


def from_numpy(array):
    """A flat 1-D float32 cpu tensor on its way out of a pipe."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).reshape(-1)


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


def apply_fault(kind, round_timeout_s):
    """Break this worker on purpose, for the demo. Runs inside the worker."""
    if kind == "sleep":
        # Wake up after the coordinator has given up on this round. The reply
        # still gets sent, late, which is exactly the stale message the pool
        # has to throw away.
        time.sleep(round_timeout_s + 0.5)
    elif kind == "exit":
        # Straight out, no cleanup and no goodbye on the pipe. The parent finds
        # out the way it would find out about a real machine going down: the
        # pipe reaches end of file.
        os._exit(1)


def worker_process(conn, cfg, worker_id, x_np, y_np, spec, faults):
    """Entry point for one worker process. Top level so spawn can pickle it.

    The process gets its data, its spec and its faults once, right here, and
    then only ever receives (round, global weights) and replies with an update.
    """
    try:
        # One thread each. Ten processes fighting over the same cores is slower
        # than one process doing the work, and it makes the timings noisy.
        torch.set_num_threads(1)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        model = make_model(len(CLASS_NAMES))
        conn.send({"kind": READY, "worker_id": worker_id})

        while True:
            msg = conn.recv()
            if msg.get("kind") == SHUTDOWN:
                break
            rnd = msg["round"]
            apply_fault(faults.get(rnd), cfg.round_timeout_s)
            global_flat = from_numpy(msg["flat"])
            try:
                delta = run_worker_round(
                    model, global_flat, x, y, spec, worker_id, rnd, cfg, "cpu"
                )
                reply = {"kind": UPDATE, "worker_id": worker_id, "round": rnd, "delta": to_numpy(delta)}
            except Exception as exc:
                # Training blew up but the process is still healthy, so say so
                # and stay alive for the next round.
                reply = {
                    "kind": UPDATE,
                    "worker_id": worker_id,
                    "round": rnd,
                    "delta": None,
                    "error": repr(exc),
                }
            conn.send(reply)
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class MultiprocessPool(WorkerPool):
    """One process per worker, one pipe per worker, no restarts.

    A round is: send (round, global weights) to everyone active, then collect
    replies against a single shared deadline. Slow workers are not given a
    fresh timeout each, or ten slow workers would hold the round for ten
    timeouts.
    """

    def __init__(self, cfg, splits, specs, startup_timeout_s=STARTUP_TIMEOUT_S):
        super().__init__(cfg, splits, specs)
        self._ctx = mp.get_context("spawn")
        self._procs = {}
        self._conns = {}
        self._owner = {}  # connection -> worker id
        self._failed = set()
        self._status = {}
        self._closed = False
        self.discarded = []  # (worker id, round) of every reply that arrived late
        try:
            self._start(startup_timeout_s)
        except Exception:
            self.close()
            raise

    def _start(self, startup_timeout_s):
        for wid in range(self.cfg.num_workers):
            x, y = self.splits[wid]
            # Each process learns its own spec and its own faults and nothing
            # about anybody else, so a worker cannot behave differently because
            # of who else is in the run.
            worker_cfg = replace(self.cfg, attackers={}, faults=[])
            faults = {
                int(f["round"]): f["kind"] for f in self.cfg.faults if f["worker"] == wid
            }
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=worker_process,
                args=(
                    child_conn,
                    worker_cfg,
                    wid,
                    x.detach().cpu().contiguous().numpy(),
                    y.detach().cpu().contiguous().numpy(),
                    self.specs.get(wid),
                    faults,
                ),
                daemon=True,
                name=f"antidote-worker-{wid}",
            )
            proc.start()
            # The parent must drop its copy of the child end. While it holds
            # one, a dead child never shows up as end of file on the pipe.
            child_conn.close()
            self._procs[wid] = proc
            self._conns[wid] = parent_conn
            self._owner[parent_conn] = wid
        self._wait_for_ready(startup_timeout_s)

    def _wait_for_ready(self, startup_timeout_s):
        """Nobody trains until every process has imported torch and built its
        model. Without this handshake, startup is charged to round 1 and every
        worker looks like it timed out."""
        waiting = {conn for conn in self._conns.values()}
        deadline = time.monotonic() + startup_timeout_s
        while waiting:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for conn in wait(list(waiting), timeout=remaining):
                wid = self._owner[conn]
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    self._mark_failed(wid)
                    waiting.discard(conn)
                    continue
                if msg.get("kind") == READY:
                    waiting.discard(conn)
        for conn in waiting:
            # Never said hello. Treat it as a machine that did not come up.
            self._mark_failed(self._owner[conn])

    def _mark_failed(self, wid):
        """Failed is forever. A machine that died stays dead for this run."""
        self._failed.add(wid)

    def run_round(self, global_flat, round, active_ids):
        results = {}
        self._status = {}
        flat_np = to_numpy(global_flat)

        pending = set()
        for wid in active_ids:
            if wid in self._failed:
                results[wid] = None
                self._status[wid] = "failed"
                continue
            try:
                self._conns[wid].send({"kind": TASK, "round": round, "flat": flat_np})
                pending.add(wid)
            except (BrokenPipeError, OSError, ValueError):
                self._mark_failed(wid)
                results[wid] = None
                self._status[wid] = "failed"

        # One deadline for the whole round, started once every task is out.
        deadline = time.monotonic() + self.cfg.round_timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready = wait([self._conns[wid] for wid in pending], timeout=remaining)
            if not ready:
                break
            for conn in ready:
                wid = self._owner[conn]
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    self._mark_failed(wid)
                    results[wid] = None
                    self._status[wid] = "failed"
                    pending.discard(wid)
                    continue
                if msg.get("round") != round:
                    # A reply from a round we already closed. Using it would
                    # add an update computed against weights that are now two
                    # rounds old, so it goes in the bin and we keep waiting.
                    self.discarded.append((wid, msg.get("round")))
                    continue
                delta = msg.get("delta")
                results[wid] = None if delta is None else from_numpy(delta)
                self._status[wid] = "failed" if delta is None else "ok"
                pending.discard(wid)

        for wid in pending:
            results[wid] = None
            self._status[wid] = "timeout"
        return results

    def status(self):
        return dict(self._status)

    def close(self):
        """Safe to call twice, and leaves no process behind either time."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns.values():
            try:
                conn.send({"kind": SHUTDOWN})
            except Exception:
                pass
        for conn in self._conns.values():
            try:
                conn.close()
            except Exception:
                pass
        for proc in self._procs.values():
            proc.join(timeout=CLOSE_TIMEOUT_S)
            if proc.is_alive():
                # Asleep, wedged, or ignoring us. Stop being polite.
                proc.terminate()
                proc.join(timeout=CLOSE_TIMEOUT_S)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=CLOSE_TIMEOUT_S)
        self._procs.clear()
        self._conns.clear()
        self._owner.clear()


def make_pool(cfg, splits, specs):
    """Pick a pool by cfg.execution."""
    if cfg.execution == "inprocess":
        return InProcessPool(cfg, splits, specs)
    if cfg.execution == "multiprocess":
        return MultiprocessPool(cfg, splits, specs)
    raise ValueError(f"unknown execution: {cfg.execution}")
