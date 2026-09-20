import pytest

import antidote.system.run as run_module


def test_cli_closes_pool_when_training_fails(monkeypatch):
    class Pool:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    pool = Pool()
    cfg = type(
        "Cfg",
        (),
        {
            "name": "broken",
            "seed": 42,
            "dataset": "gtsrb",
            "num_workers": 2,
            "attackers": {},
        },
    )()
    monkeypatch.setattr(run_module, "load_config", lambda path: cfg)
    monkeypatch.setattr(run_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_module, "pick_device", lambda: "cpu")
    monkeypatch.setattr(run_module, "load_data", lambda *args: ([], None, None))
    monkeypatch.setattr(run_module, "make_pool", lambda *args: pool)
    monkeypatch.setattr(
        run_module,
        "run_training",
        lambda *args: (_ for _ in ()).throw(RuntimeError("training failed")),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        run_module.main(["--config", "configs/broken.yaml"])

    assert pool.closed is True
