from pathlib import Path

import yaml

from antidote.system.config import load_config


CONFIG_DIR = Path(__file__).parents[1] / "configs"
SCENARIOS = {
    "clean_off": (False, None),
    "clean_on": (True, None),
    "backdoor_off": (False, 6),
    "backdoor_on": (True, 6),
    "faulty_off": (False, 2),
    "faulty_on": (True, 2),
}
REQUIRED_KEYS = {
    "name",
    "seed",
    "dataset",
    "num_workers",
    "rounds",
    "local_epochs",
    "batch_size",
    "lr",
    "momentum",
    "target_class",
    "defense",
    "aggregator",
    "trim_frac",
    "detector",
    "execution",
}


def test_all_scenario_configs_have_the_frozen_shape() -> None:
    for name, (defense, attacker_id) in SCENARIOS.items():
        config = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())

        assert REQUIRED_KEYS <= config.keys()
        assert config["name"] == name
        assert config["defense"] == defense
        assert config["aggregator"] == (
            "trimmed_mean" if defense else "mean"
        )

        if attacker_id is None:
            assert "attackers" not in config
        else:
            assert set(config["attackers"]) == {attacker_id}


def test_backdoor_and_faulty_scenarios_use_distinct_workers() -> None:
    backdoor = yaml.safe_load((CONFIG_DIR / "backdoor_off.yaml").read_text())
    faulty = yaml.safe_load((CONFIG_DIR / "faulty_off.yaml").read_text())

    backdoor_id = next(iter(backdoor["attackers"]))
    faulty_id = next(iter(faulty["attackers"]))
    assert backdoor_id != faulty_id
    assert backdoor["attackers"][backdoor_id]["mode"] == "scale"
    assert faulty["attackers"][faulty_id]["mode"] == "noise"


def test_all_scenario_configs_load_through_system_parser() -> None:
    for name, (defense, _) in SCENARIOS.items():
        config = load_config(CONFIG_DIR / f"{name}.yaml")
        assert config.name == name
        assert config.defense is defense
