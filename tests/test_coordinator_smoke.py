import json

from antidote.ml import load_data
from antidote.system.config import load_config
from antidote.system.coordinator import run_training
from antidote.system.pool import make_pool


def test_smoke_config_runs_end_to_end_and_writes_valid_jsonl(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    splits, x_test, y_test = load_data(cfg.dataset, cfg.num_workers, cfg.seed)
    pool = make_pool(cfg, splits, cfg.attackers)
    path = tmp_path / "smoke.jsonl"
    try:
        final = run_training(cfg, pool, splits, x_test, y_test, path)
    finally:
        pool.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert final.ndim == 1
    assert records[0]["type"] == "header"
    assert [record["round"] for record in records[1:-1]] == [1, 2, 3]
    assert records[-1]["type"] == "summary"
