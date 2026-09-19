"""Config loading, seeding and device choice. Owner: Person 4.

Minimal version for the skeleton. Step 2 adds full validation.
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
    """Read a YAML scenario file into a Config."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in Config.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**raw)
    cfg.detector = {**DEFAULT_DETECTOR, **(cfg.detector or {})}
    cfg.attackers = {int(k): v for k, v in (cfg.attackers or {}).items()}
    cfg.faults = list(cfg.faults or [])
    validate(cfg)
    return cfg


def validate(cfg):
    """Cheap checks that catch a typo before a long run starts."""
    if cfg.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if cfg.execution not in ("inprocess", "multiprocess"):
        raise ValueError(f"unknown execution: {cfg.execution}")
    if cfg.aggregator not in ("mean", "median", "trimmed_mean"):
        raise ValueError(f"unknown aggregator: {cfg.aggregator}")
    for wid in cfg.attackers:
        if not 0 <= wid < cfg.num_workers:
            raise ValueError(f"attacker id {wid} out of range")
    for fault in cfg.faults:
        wid = fault.get("worker")
        if not 0 <= wid < cfg.num_workers:
            raise ValueError(f"fault worker id {wid} out of range")
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
