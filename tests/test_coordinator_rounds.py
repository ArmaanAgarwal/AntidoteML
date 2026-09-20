import json
from types import SimpleNamespace

import torch
import pytest

import antidote.system.coordinator as coordinator_module
from antidote.types import Score


def make_cfg(*, defense=True, rounds=1, num_workers=3, attackers=None):
    return SimpleNamespace(
        name="test",
        seed=42,
        dataset="gtsrb",
        num_workers=num_workers,
        rounds=rounds,
        local_epochs=1,
        batch_size=4,
        lr=0.01,
        momentum=0.9,
        target_class=5,
        defense=defense,
        aggregator="mean" if not defense else "trimmed_mean",
        trim_frac=0.2,
        detector={
            "warmup": 0,
            "threshold": 3.5,
            "strikes": 1,
            "window": 1,
            "hard_norm_ratio": 5.0,
        },
        execution="inprocess",
        round_timeout_s=1,
        train_images=10,
        attackers=attackers or {},
        faults=[],
    )


class FakePool:
    def __init__(self, responses, statuses=None):
        self.responses = responses
        self.statuses = statuses or [{} for _ in responses]
        self.calls = []
        self.index = 0

    def run_round(self, global_flat, round, active_ids):
        self.calls.append((round, list(active_ids)))
        response = self.responses[self.index]
        self.index += 1
        return response

    def status(self):
        return self.statuses[self.index - 1]


def install_flat_model(monkeypatch):
    monkeypatch.setattr(coordinator_module, "make_model", lambda _: object())
    monkeypatch.setattr(coordinator_module, "get_flat", lambda _: torch.zeros(1))
    monkeypatch.setattr(
        coordinator_module,
        "evaluate",
        lambda flat, x, y, target, device: (float(flat.item()), 0.0),
    )


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_config_without_faults_logs_worker_as_honest(monkeypatch, tmp_path):
    install_flat_model(monkeypatch)
    cfg = make_cfg(defense=False, num_workers=1)
    delattr(cfg, "faults")
    pool = FakePool([{0: torch.tensor([1.0])}])
    path = tmp_path / "run.jsonl"

    coordinator_module.run_training(cfg, pool, [], None, None, path)

    round_record = read_records(path)[1]
    assert round_record["workers"][0]["role"] == "honest"


def test_pool_without_status_uses_failed_fallback(monkeypatch, tmp_path):
    install_flat_model(monkeypatch)

    class PoolWithoutStatus:
        def run_round(self, global_flat, round, active_ids):
            return {0: torch.tensor([1.0]), 1: None}

    path = tmp_path / "run.jsonl"

    coordinator_module.run_training(
        make_cfg(defense=False, num_workers=2),
        PoolWithoutStatus(),
        [],
        None,
        None,
        path,
    )

    records = read_records(path)
    assert [record["type"] for record in records] == ["header", "round", "summary"]
    worker_one = next(row for row in records[1]["workers"] if row["id"] == 1)
    assert worker_one["participated"] is False
    assert any(
        event["type"] == "failed" and event["worker_id"] == 1
        for event in records[1]["events"]
    )


def test_pool_status_treats_not_implemented_as_empty():
    class PoolWithoutStatusImplementation:
        def status(self):
            raise NotImplementedError

    assert coordinator_module._pool_status(PoolWithoutStatusImplementation()) == {}


def test_pool_status_propagates_other_errors():
    class BrokenPool:
        def status(self):
            raise RuntimeError("status unavailable")

    with pytest.raises(RuntimeError, match="status unavailable"):
        coordinator_module._pool_status(BrokenPool())


def test_none_results_are_not_scored_and_are_logged(monkeypatch, tmp_path):
    install_flat_model(monkeypatch)
    seen_ids = []

    class Detector:
        def __init__(self, **kwargs):
            pass

        def inspect(self, deltas, worker_ids, round):
            seen_ids.extend(worker_ids)
            return [
                Score(wid, 1.0, 1.0, 0.0, 1.0, "trusted", "")
                for wid in worker_ids
            ]

        def ejected(self):
            return set()

    monkeypatch.setattr(coordinator_module, "Detector", Detector)
    pool = FakePool(
        [{0: torch.tensor([1.0]), 1: None}],
        [{0: "ok", 1: "timeout"}],
    )
    path = tmp_path / "run.jsonl"

    coordinator_module.run_training(
        make_cfg(num_workers=2), pool, [], None, None, path
    )

    assert seen_ids == [0]
    round_record = read_records(path)[1]
    worker_one = next(row for row in round_record["workers"] if row["id"] == 1)
    assert worker_one["participated"] is False
    assert any(
        event["type"] == "timeout" and event["worker_id"] == 1
        for event in round_record["events"]
    )


def test_non_finite_update_is_dropped_when_defense_is_off(monkeypatch, tmp_path):
    install_flat_model(monkeypatch)
    monkeypatch.setattr(
        coordinator_module,
        "compute_features",
        lambda deltas: [
            {
                "norm_ratio": None if not torch.isfinite(row).all() else 1.0,
                "cosine": None if not torch.isfinite(row).all() else 1.0,
            }
            for row in deltas
        ],
    )
    pool = FakePool(
        [{0: torch.tensor([2.0]), 1: torch.tensor([float("nan")])}]
    )
    path = tmp_path / "run.jsonl"

    final = coordinator_module.run_training(
        make_cfg(defense=False, num_workers=2), pool, [], None, None, path
    )

    assert torch.equal(final, torch.tensor([2.0]))
    round_record = read_records(path)[1]
    bad_row = next(row for row in round_record["workers"] if row["id"] == 1)
    assert bad_row["participated"] is True
    assert bad_row["kept"] is False
    assert bad_row["norm_ratio"] is None


def test_safety_net_keeps_all_finite_updates_when_too_few_are_trusted(
    monkeypatch, tmp_path
):
    install_flat_model(monkeypatch)

    class Detector:
        def __init__(self, **kwargs):
            pass

        def inspect(self, deltas, worker_ids, round):
            return [
                Score(wid, 1.0, 1.0, float(wid), 1.0, "trusted" if wid == 0 else "suspect", "flagged")
                for wid in worker_ids
            ]

        def ejected(self):
            return set()

    monkeypatch.setattr(coordinator_module, "Detector", Detector)
    pool = FakePool(
        [
            {
                0: torch.tensor([1.0]),
                1: torch.tensor([2.0]),
                2: torch.tensor([3.0]),
            }
        ]
    )
    path = tmp_path / "run.jsonl"

    final = coordinator_module.run_training(
        make_cfg(), pool, [], None, None, path
    )

    assert torch.equal(final, torch.tensor([2.0]))
    round_record = read_records(path)[1]
    assert all(row["kept"] for row in round_record["workers"])
    assert any(event["type"] == "safety_net" for event in round_record["events"])


def test_ejected_worker_is_not_requested_again(monkeypatch, tmp_path):
    install_flat_model(monkeypatch)

    class Detector:
        def __init__(self, **kwargs):
            self._ejected = set()

        def inspect(self, deltas, worker_ids, round):
            scores = []
            for wid in worker_ids:
                if round == 1 and wid == 1:
                    self._ejected.add(1)
                    scores.append(
                        Score(1, 8.0, -1.0, 10.0, 0.7, "ejected", "outlier")
                    )
                else:
                    scores.append(Score(wid, 1.0, 1.0, 0.0, 1.0, "trusted", ""))
            return scores

        def ejected(self):
            return set(self._ejected)

    monkeypatch.setattr(coordinator_module, "Detector", Detector)
    pool = FakePool(
        [
            {0: torch.tensor([1.0]), 1: torch.tensor([9.0])},
            {0: torch.tensor([1.0])},
        ]
    )
    path = tmp_path / "run.jsonl"

    coordinator_module.run_training(
        make_cfg(rounds=2, num_workers=2), pool, [], None, None, path
    )

    assert pool.calls == [(1, [0, 1]), (2, [0])]
    summary = read_records(path)[-1]
    assert summary["ejected"][0]["worker_id"] == 1
    assert summary["false_positives"] == 1


def test_attack_started_event_and_ground_truth_role_are_log_only(
    monkeypatch, tmp_path
):
    install_flat_model(monkeypatch)
    pool = FakePool([{0: torch.tensor([1.0]), 1: torch.tensor([2.0])}])
    path = tmp_path / "run.jsonl"
    cfg = make_cfg(
        defense=False,
        num_workers=2,
        attackers={1: {"mode": "scale", "start_round": 1}},
    )

    coordinator_module.run_training(cfg, pool, [], None, None, path)

    round_record = read_records(path)[1]
    attacker = next(row for row in round_record["workers"] if row["id"] == 1)
    assert attacker["role"] == "backdoor"
    assert any(event["type"] == "attack_started" for event in round_record["events"])


def test_log_closes_if_header_write_fails(monkeypatch, tmp_path):
    install_flat_model(monkeypatch)

    class BrokenLog:
        def __init__(self, path):
            self.closed = False

        def write(self, record):
            raise OSError("disk full")

        def close(self):
            self.closed = True

    broken_log = BrokenLog(tmp_path / "run.jsonl")
    monkeypatch.setattr(coordinator_module, "RunLog", lambda path: broken_log)
    pool = FakePool([{0: torch.tensor([1.0])}])

    with pytest.raises(OSError, match="disk full"):
        coordinator_module.run_training(
            make_cfg(defense=False, num_workers=1),
            pool,
            [],
            None,
            None,
            tmp_path / "run.jsonl",
        )

    assert broken_log.closed is True
