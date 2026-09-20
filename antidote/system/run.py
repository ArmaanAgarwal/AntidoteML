"""Command line entry point for AntidoteML. Owner: Person 3.

Run with: python -m antidote.system.run --config configs/smoke.yaml
"""

import argparse

from antidote.ml import load_data
from antidote.system.config import load_config, pick_device, set_seed
from antidote.system.coordinator import run_training
from antidote.system.pool import make_pool


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an AntidoteML scenario")
    parser.add_argument("--config", required=True, help="path to a scenario YAML file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = pick_device()
    log_path = f"runs/{cfg.name}.jsonl"
    print(f"run {cfg.name} on {device}")

    splits, x_test, y_test = load_data(cfg.dataset, cfg.num_workers, cfg.seed)
    pool = make_pool(cfg, splits, cfg.attackers)
    try:
        run_training(
            cfg,
            pool,
            splits,
            x_test,
            y_test,
            log_path,
            device,
        )
    finally:
        pool.close()
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
