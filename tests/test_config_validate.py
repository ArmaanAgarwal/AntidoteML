"""Person 4 tests for config loading and validation.

A bad config should fail loudly in the first second of a run, not silently
turn into a run where the attacker attacks nobody.
"""

import pytest
import torch
import yaml

from antidote.system.config import (
    DEFAULT_DETECTOR,
    Config,
    load_config,
    pick_device,
    set_seed,
    validate,
)

BASE = {
    "name": "unit",
    "seed": 7,
    "dataset": "gtsrb",
    "num_workers": 5,
    "rounds": 3,
    "local_epochs": 1,
    "batch_size": 8,
    "lr": 0.01,
    "momentum": 0.9,
    "target_class": 5,
    "train_images": 100,
    "defense": False,
    "aggregator": "mean",
    "trim_frac": 0.2,
    "execution": "inprocess",
    "round_timeout_s": 5,
}


def write(tmp_path, **overrides):
    raw = {**BASE, **overrides}
    for key, value in list(raw.items()):
        if value is None:
            del raw[key]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_loads_a_good_config(tmp_path):
    cfg = load_config(write(tmp_path))
    assert isinstance(cfg, Config)
    assert cfg.name == "unit"
    assert cfg.seed == 7
    assert cfg.num_workers == 5
    assert cfg.train_images == 100
    assert cfg.execution == "inprocess"
    assert cfg.round_timeout_s == 5
    assert cfg.attackers == {}
    assert cfg.faults == []


def test_detector_block_fills_in_defaults(tmp_path):
    cfg = load_config(write(tmp_path, detector={"threshold": 9.0}))
    assert cfg.detector["threshold"] == 9.0
    assert cfg.detector["warmup"] == DEFAULT_DETECTOR["warmup"]
    assert set(cfg.detector) == set(DEFAULT_DETECTOR)


def test_unknown_detector_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="detector"):
        load_config(write(tmp_path, detector={"treshold": 9.0}))


def test_attacker_ids_become_ints(tmp_path):
    cfg = load_config(write(tmp_path, attackers={3: {"mode": "scale", "start_round": 2}}))
    assert list(cfg.attackers) == [3]
    assert cfg.attackers[3]["mode"] == "scale"


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(write(tmp_path, rouds=3))


@pytest.mark.parametrize("bad", [5, 9, -1])
def test_attacker_id_out_of_range_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="attacker"):
        load_config(write(tmp_path, attackers={bad: {"mode": "scale", "start_round": 2}}))


def test_unknown_attacker_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        load_config(write(tmp_path, attackers={2: {"mode": "scaleup", "start_round": 2}}))


def test_fault_worker_out_of_range_rejected(tmp_path):
    with pytest.raises(ValueError, match="fault"):
        load_config(
            write(
                tmp_path,
                execution="multiprocess",
                faults=[{"worker": 12, "round": 2, "kind": "sleep"}],
            )
        )


def test_unknown_fault_kind_rejected(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        load_config(
            write(
                tmp_path,
                execution="multiprocess",
                faults=[{"worker": 1, "round": 2, "kind": "explode"}],
            )
        )


def test_faults_need_the_multiprocess_pool(tmp_path):
    with pytest.raises(ValueError, match="multiprocess"):
        load_config(
            write(
                tmp_path,
                execution="inprocess",
                faults=[{"worker": 1, "round": 2, "kind": "sleep"}],
            )
        )


def test_good_faults_load(tmp_path):
    cfg = load_config(
        write(
            tmp_path,
            execution="multiprocess",
            faults=[{"worker": 1, "round": 2, "kind": "sleep"}],
        )
    )
    assert cfg.faults == [{"worker": 1, "round": 2, "kind": "sleep"}]


def test_unknown_execution_rejected(tmp_path):
    with pytest.raises(ValueError, match="execution"):
        load_config(write(tmp_path, execution="threads"))


def test_unknown_aggregator_rejected(tmp_path):
    with pytest.raises(ValueError, match="aggregator"):
        load_config(write(tmp_path, aggregator="avg"))


def test_unknown_dataset_rejected(tmp_path):
    with pytest.raises(ValueError, match="dataset"):
        load_config(write(tmp_path, dataset="cifar"))


def test_defense_must_be_a_boolean(tmp_path):
    with pytest.raises(ValueError, match="defense"):
        load_config(write(tmp_path, defense="off"))


@pytest.mark.parametrize("bad", [-0.1, 0.5, 0.9])
def test_trim_frac_out_of_range_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="trim_frac"):
        load_config(write(tmp_path, trim_frac=bad))


def test_train_images_must_cover_every_worker(tmp_path):
    with pytest.raises(ValueError, match="train_images"):
        load_config(write(tmp_path, num_workers=5, train_images=3))


def test_train_images_may_be_absent(tmp_path):
    cfg = load_config(write(tmp_path, train_images=None))
    assert cfg.train_images is None


def test_round_timeout_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="round_timeout_s"):
        load_config(write(tmp_path, round_timeout_s=0))


def test_validate_returns_the_config():
    cfg = Config(name="unit", num_workers=3)
    assert validate(cfg) is cfg


def test_set_seed_makes_torch_repeatable():
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    assert torch.equal(first, torch.rand(4))


def test_pick_device_is_one_of_three():
    assert pick_device() in ("mps", "cuda", "cpu")
