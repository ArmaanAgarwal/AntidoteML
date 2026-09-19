"""Person 4 tests for the scenario configs in configs/.

Every file in configs/ has to load and validate, and each off and on pair has
to be the same run with the defense switched, otherwise the two plots we put in
front of a judge are not comparing the same thing.
"""

import glob
import os

import pytest
import yaml

from antidote.system.config import load_config

SCENARIOS = [
    "clean_off",
    "clean_on",
    "backdoor_off",
    "backdoor_on",
    "faulty_off",
    "faulty_on",
]
PAIRS = [("clean_off", "clean_on"), ("backdoor_off", "backdoor_on"), ("faulty_off", "faulty_on")]


def path(name):
    return os.path.join("configs", f"{name}.yaml")


def raw(name):
    with open(path(name)) as f:
        return yaml.safe_load(f)


def config_files():
    return sorted(glob.glob(os.path.join("configs", "*.yaml")))


def test_every_scenario_file_exists():
    missing = [name for name in SCENARIOS + ["smoke"] if not os.path.exists(path(name))]
    assert missing == []


@pytest.mark.parametrize("file", config_files())
def test_every_config_in_the_folder_loads_and_validates(file):
    cfg = load_config(file)
    assert cfg.name == os.path.splitext(os.path.basename(file))[0]


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_file_name_matches_the_defense_switch(name):
    cfg = load_config(path(name))
    assert cfg.defense is name.endswith("_on")


@pytest.mark.parametrize("name", SCENARIOS)
def test_defense_off_means_a_plain_mean(name):
    cfg = load_config(path(name))
    assert cfg.aggregator == ("trimmed_mean" if cfg.defense else "mean")


@pytest.mark.parametrize("off,on", PAIRS)
def test_each_pair_differs_only_by_the_defense(off, on):
    a, b = raw(off), raw(on)
    for key in ("name", "defense", "aggregator"):
        a.pop(key, None)
        b.pop(key, None)
    assert a == b


def test_the_clean_scenarios_have_no_attacker():
    for name in ("clean_off", "clean_on"):
        assert load_config(path(name)).attackers == {}


def test_the_backdoor_scenario_scales_a_poisoned_update():
    for name in ("backdoor_off", "backdoor_on"):
        attackers = load_config(path(name)).attackers
        assert len(attackers) == 1
        spec = next(iter(attackers.values()))
        assert spec["mode"] == "scale"
        assert spec["start_round"] >= 1
        assert 0 < spec["poison_frac"] <= 1
        assert spec["scale"] > 1


def test_the_faulty_scenario_sends_noise():
    for name in ("faulty_off", "faulty_on"):
        attackers = load_config(path(name)).attackers
        assert len(attackers) == 1
        spec = next(iter(attackers.values()))
        assert spec["mode"] == "noise"


def test_the_backdoor_and_the_faulty_worker_are_different_machines():
    backdoor = set(load_config(path("backdoor_on")).attackers)
    faulty = set(load_config(path("faulty_on")).attackers)
    assert backdoor and faulty
    assert backdoor.isdisjoint(faulty)


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_attack_starts_after_the_detector_warmup(name):
    cfg = load_config(path(name))
    for wid, spec in cfg.attackers.items():
        assert spec["start_round"] > cfg.detector["warmup"]
        assert spec["start_round"] < cfg.rounds


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_scenario_runs_the_same_model_and_seed(name):
    cfg = load_config(path(name))
    assert cfg.seed == 42
    assert cfg.num_workers == 10
    assert cfg.rounds == 40
    assert cfg.dataset == "gtsrb"
