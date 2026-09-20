import json

from antidote.system.runlog import RunLog


def test_runlog_flushes_each_line_and_replaces_non_finite_values(tmp_path):
    path = tmp_path / "nested" / "run.jsonl"
    log = RunLog(path)
    log.write(
        {
            "type": "round",
            "clean_acc": float("nan"),
            "workers": [{"anomaly": float("inf")}],
        }
    )

    record = json.loads(path.read_text().strip())
    assert record["clean_acc"] is None
    assert record["workers"][0]["anomaly"] is None
    log.close()


def test_runlog_close_is_idempotent(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    log.close()
    log.close()
