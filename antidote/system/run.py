"""PLACEHOLDER CLI for Person 3.

python -m antidote.system.run --config configs/smoke.yaml
"""

import argparse

from antidote.ml import load_data
from antidote.system.config import load_config, pick_device, set_seed
from antidote.system.coordinator import run_training
from antidote.system.pool import make_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = pick_device()
    print(f"run {cfg.name} on {device}")

    splits, x_test, y_test = load_data(cfg.dataset, cfg.num_workers, cfg.seed)
    pool = make_pool(cfg, splits, cfg.attackers)
    try:
        run_training(cfg, pool, splits, x_test, y_test, f"runs/{cfg.name}.jsonl", device)
    finally:
        pool.close()
    print(f"wrote runs/{cfg.name}.jsonl")


if __name__ == "__main__":
    main()
