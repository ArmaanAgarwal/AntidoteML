"""Config loading, seeding and device choice. Owner: Person 4.

One YAML scenario file becomes one Config dataclass. Everything cheap to check
is checked here, before a run starts, because a typo in a worker id reads much
better as an error message than as a forty round run that quietly attacked
nobody.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

DEFAULT_DETECTOR = {
    "warmup": 3,
    "threshold": 3.5,
    "strikes": 3,
    "window": 5,
    "hard_norm_ratio": 5.0,
}

DATASETS = ("gtsrb", "mnist")
AGGREGATORS = ("mean", "median", "trimmed_mean")
EXECUTIONS = ("inprocess", "multiprocess")
ATTACK_MODES = ("scale", "noise", "nan")
FAULT_KINDS = ("sleep", "exit")


@dataclass
class Config:
    name: str
    seed: int = 42
    dataset: str = "gtsrb"
    num_workers: int = 10
    rounds: int = 40
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    target_class: int = 5
    defense: bool = False
    aggregator: str = "mean"
    trim_frac: float = 0.2
    detector: dict = field(default_factory=lambda: dict(DEFAULT_DETECTOR))
    execution: str = "inprocess"
    round_timeout_s: float = 60.0
    train_images: int | None = None
    attackers: dict = field(default_factory=dict)
    faults: list = field(default_factory=list)


def load_config(path):
    """Read a YAML scenario file into a validated Config."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping")

    known = set(Config.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")

    cfg = Config(**raw)
    cfg.detector = {**DEFAULT_DETECTOR, **(cfg.detector or {})}
    cfg.attackers = {int(k): dict(v or {}) for k, v in (cfg.attackers or {}).items()}
    cfg.faults = [dict(f) for f in (cfg.faults or [])]
    validate(cfg)
    return cfg


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def validate(cfg):
    """Raise ValueError on anything a run would only discover the slow way."""
    _check(isinstance(cfg.name, str) and cfg.name != "", "name must be a non empty string")
    _check(cfg.dataset in DATASETS, f"unknown dataset {cfg.dataset!r}, pick one of {list(DATASETS)}")
    _check(cfg.num_workers >= 1, "num_workers must be at least 1")
    _check(cfg.rounds >= 1, "rounds must be at least 1")
    _check(cfg.local_epochs >= 1, "local_epochs must be at least 1")
    _check(cfg.batch_size >= 1, "batch_size must be at least 1")
    _check(cfg.lr > 0, "lr must be positive")
    _check(0 <= cfg.momentum < 1, "momentum must be in [0, 1)")
    _check(cfg.target_class >= 0, "target_class must be zero or more")

    # yaml turns bare on and off into booleans, but "off" in quotes stays a
    # truthy string, which would silently run the defence in an off scenario.
    _check(isinstance(cfg.defense, bool), "defense must be a boolean, write on or off unquoted")

    _check(
        cfg.aggregator in AGGREGATORS,
        f"unknown aggregator {cfg.aggregator!r}, pick one of {list(AGGREGATORS)}",
    )
    _check(0 <= cfg.trim_frac < 0.5, "trim_frac must be in [0, 0.5)")
    _check(
        cfg.execution in EXECUTIONS,
        f"unknown execution {cfg.execution!r}, pick one of {list(EXECUTIONS)}",
    )
    _check(cfg.round_timeout_s > 0, "round_timeout_s must be positive")

    unknown_detector = set(cfg.detector) - set(DEFAULT_DETECTOR)
    _check(not unknown_detector, f"unknown detector keys: {sorted(unknown_detector)}")

    if cfg.train_images is not None:
        _check(cfg.train_images >= 1, "train_images must be positive or absent")
        _check(
            cfg.train_images // cfg.num_workers >= 1,
            f"train_images {cfg.train_images} leaves nothing for each of {cfg.num_workers} workers",
        )

    for wid, spec in cfg.attackers.items():
        _check(
            0 <= wid < cfg.num_workers,
            f"attacker id {wid} is outside 0..{cfg.num_workers - 1}",
        )
        _check(isinstance(spec, dict), f"attacker {wid} spec must be a mapping")
        mode = spec.get("mode")
        _check(
            mode in ATTACK_MODES,
            f"attacker {wid} has unknown mode {mode!r}, pick one of {list(ATTACK_MODES)}",
        )
        _check(int(spec.get("start_round", 1)) >= 1, f"attacker {wid} start_round must be at least 1")

    if cfg.faults:
        # A fault is a machine that hangs or dies. Only the multiprocess pool
        # has real processes to hang or kill, so asking for one anywhere else
        # is a mistake worth saying out loud.
        _check(
            cfg.execution == "multiprocess",
            "faults only work with execution multiprocess",
        )
    for fault in cfg.faults:
        _check(isinstance(fault, dict), "each fault must be a mapping")
        wid = fault.get("worker")
        _check(
            isinstance(wid, int) and 0 <= wid < cfg.num_workers,
            f"fault worker id {wid} is outside 0..{cfg.num_workers - 1}",
        )
        _check(int(fault.get("round", 0)) >= 1, f"fault on worker {wid} needs a round of 1 or more")
        kind = fault.get("kind")
        _check(
            kind in FAULT_KINDS,
            f"fault on worker {wid} has unknown kind {kind!r}, pick one of {list(FAULT_KINDS)}",
        )
    return cfg


def set_seed(seed):
    """Seed every generator the run touches. All randomness starts here."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def pick_device():
    """Apple GPU, then nvidia GPU, then cpu. Workers always stay on cpu."""
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
