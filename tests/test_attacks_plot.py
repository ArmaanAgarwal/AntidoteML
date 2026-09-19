import json
from pathlib import Path

import pytest

from experiments.plot_run import load_run, plot_runs


def _write_log(path: Path, name: str, asr: float) -> None:
    rows = [
        {
            "type": "header",
            "name": name,
            "config": {
                "attackers": {
                    "6": {
                        "mode": "scale",
                        "start_round": 2,
                        "poison_frac": 0.5,
                        "scale": 10,
                    }
                }
            },
            "class_names": [],
        },
        {
            "type": "round",
            "round": 1,
            "clean_acc": 0.8,
            "asr": 0.0,
            "workers": [
                {"id": 0, "norm_ratio": 1.0},
                {"id": 6, "norm_ratio": 1.1},
            ],
            "events": [],
        },
        {
            "type": "round",
            "round": 2,
            "clean_acc": 0.9,
            "asr": asr,
            "workers": [
                {"id": 0, "norm_ratio": 0.9},
                {"id": 6, "norm_ratio": 9.7},
            ],
            "events": [],
        },
        {
            "type": "summary",
            "final_clean_acc": 0.9,
            "final_asr": asr,
            "ejected": [{"worker_id": 6, "round": 3, "reason": "large update"}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_run_extracts_header_rounds_and_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    _write_log(log_path, "backdoor_off", 0.85)

    run = load_run(log_path)

    assert run["header"]["name"] == "backdoor_off"
    assert [row["round"] for row in run["rounds"]] == [1, 2]
    assert run["summary"]["final_asr"] == 0.85


def test_load_run_rejects_missing_header(tmp_path: Path) -> None:
    log_path = tmp_path / "bad.jsonl"
    log_path.write_text('{"type": "round", "round": 1}\n')

    with pytest.raises(ValueError, match="header"):
        load_run(log_path)


def test_plot_runs_saves_png_for_one_or_two_logs(tmp_path: Path) -> None:
    first = tmp_path / "off.jsonl"
    second = tmp_path / "on.jsonl"
    _write_log(first, "backdoor_off", 0.85)
    _write_log(second, "backdoor_on", 0.05)

    one_output = plot_runs([first], tmp_path / "one.png")
    two_output = plot_runs([first, second], tmp_path / "two.png")

    for output in (one_output, two_output):
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert output.stat().st_size > 1_000


def test_plot_runs_accepts_at_most_two_logs(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.jsonl" for index in range(3)]
    for path in paths:
        _write_log(path, path.stem, 0.5)

    with pytest.raises(ValueError, match="one or two"):
        plot_runs(paths)
