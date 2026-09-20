"""Person 4 tests for experiments/plot_run.py.

The plot script is tested against a hand written log so it can be trusted
before any real run exists, and so a broken log never gets discovered for the
first time while somebody is pointing a camera at the demo.
"""

import json
import math
import sys
from pathlib import Path

import pytest

# plot_run is a script, not part of the antidote package, so point at its folder
# rather than relying on which directory pytest was started from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import plot_run  # noqa: E402

ATTACKER = 2
START = 3
EJECT = 5


def worker_row(wid, norm_ratio, participated=True, role="honest"):
    if not participated:
        return {
            "id": wid,
            "role": role,
            "participated": False,
            "kept": False,
            "norm_ratio": None,
            "cosine": None,
            "anomaly": 0.0,
            "reputation": 1.0,
            "status": "ejected" if wid == ATTACKER else "trusted",
            "reason": "",
        }
    return {
        "id": wid,
        "role": role,
        "participated": True,
        "kept": True,
        "norm_ratio": norm_ratio,
        "cosine": 0.8,
        "anomaly": 0.3,
        "reputation": 1.0,
        "status": "trusted",
        "reason": "",
    }


def fake_log(path, name="fake_on", attackers=True, rounds=6):
    """Six rounds, four workers, one of them compromised from round 3."""
    config = {
        "name": name,
        "seed": 42,
        "num_workers": 4,
        "rounds": rounds,
        "defense": True,
        "attackers": {str(ATTACKER): {"mode": "scale", "start_round": START, "scale": 10}}
        if attackers
        else {},
    }
    def role_of(wid):
        return "backdoor" if attackers and wid == ATTACKER else "honest"

    lines = [{"type": "header", "name": name, "config": config, "class_names": ["a", "b"]}]
    for rnd in range(1, rounds + 1):
        workers = []
        for wid in range(4):
            gone = attackers and wid == ATTACKER and rnd > EJECT
            missing = wid == 3 and rnd == 2
            if gone or missing:
                workers.append(worker_row(wid, None, participated=False, role=role_of(wid)))
            elif attackers and wid == ATTACKER and rnd >= START:
                workers.append(worker_row(wid, 9.5, role=role_of(wid)))
            else:
                workers.append(worker_row(wid, 1.0 + 0.02 * wid, role=role_of(wid)))
        events = []
        if attackers and rnd == START:
            events.append({"type": "attack_started", "round": rnd})
        if attackers and rnd == EJECT:
            events.append(
                {"type": "eject", "worker_id": ATTACKER, "message": "worker 2 ejected"}
            )
        lines.append(
            {
                "type": "round",
                "round": rnd,
                "clean_acc": 0.5 + 0.05 * rnd,
                "asr": 0.02 if not attackers else min(0.9, 0.05 * rnd),
                "workers": workers,
                "events": events,
            }
        )
    lines.append(
        {
            "type": "summary",
            "final_clean_acc": 0.9,
            "final_asr": 0.03,
            "ejected": [{"worker_id": ATTACKER, "round": EJECT, "reason": "too big"}]
            if attackers
            else [],
            "false_positives": 0,
            "seconds": 12,
        }
    )
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return str(path)


@pytest.fixture
def log(tmp_path):
    return fake_log(tmp_path / "fake_on.jsonl")


def is_png(path):
    with open(path, "rb") as f:
        return f.read(4) == b"\x89PNG"


def test_load_run_reads_the_three_kinds_of_line(log):
    run = plot_run.load_run(log)
    assert run.name == "fake_on"
    assert len(run.rounds) == 6
    assert run.summary["false_positives"] == 0
    assert plot_run.round_numbers(run) == [1, 2, 3, 4, 5, 6]


def test_the_two_headline_metrics_come_out_per_round(log):
    run = plot_run.load_run(log)
    assert plot_run.metric(run, "clean_acc")[0] == pytest.approx(0.55)
    assert plot_run.metric(run, "asr")[-1] == pytest.approx(0.3)


def test_the_attacker_is_named_by_the_config_and_the_roles(log):
    assert plot_run.attacker_ids(plot_run.load_run(log)) == {ATTACKER}


def test_the_attack_start_is_found(log):
    assert plot_run.attack_start(plot_run.load_run(log)) == START


def test_the_attack_start_falls_back_to_the_config(tmp_path):
    path = fake_log(tmp_path / "no_events.jsonl")
    lines = [json.loads(l) for l in open(path)]
    for line in lines:
        line.pop("events", None)
    (tmp_path / "no_events.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    assert plot_run.attack_start(plot_run.load_run(path)) == START


def test_the_ejection_round_is_found(log):
    assert plot_run.ejections(plot_run.load_run(log)) == [(ATTACKER, EJECT)]


def test_a_worker_that_did_not_take_part_leaves_a_gap_not_a_zero(log):
    ratios = plot_run.norm_ratios(plot_run.load_run(log))
    assert sorted(ratios) == [0, 1, 2, 3]
    assert math.isnan(ratios[3][1])  # worker 3 sat out round 2
    assert ratios[3][0] == pytest.approx(1.06)
    assert all(math.isnan(v) for v in ratios[ATTACKER][EJECT:])  # ejected, no scores


def test_one_log_saves_a_png(tmp_path, log):
    out = tmp_path / "one.png"
    plot_run.plot_runs([plot_run.load_run(log)], str(out))
    assert is_png(out)
    assert out.stat().st_size > 5000


def test_two_logs_save_one_png_with_a_column_each(tmp_path):
    off = fake_log(tmp_path / "fake_off.jsonl", name="fake_off")
    on = fake_log(tmp_path / "fake_on.jsonl", name="fake_on")
    runs = [plot_run.load_run(off), plot_run.load_run(on)]
    out = tmp_path / "two.png"
    fig = plot_run.plot_runs(runs, str(out))
    assert is_png(out)
    assert len(fig.axes) == 4  # two panels for each run


def test_the_attacker_line_is_called_out_by_id(tmp_path, log):
    fig = plot_run.plot_runs([plot_run.load_run(log)], str(tmp_path / "x.png"))
    labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
    assert any("2" in str(label) and "worker" in str(label).lower() for label in labels)


def test_a_clean_run_with_no_attacker_still_plots(tmp_path):
    path = fake_log(tmp_path / "fake_clean.jsonl", name="fake_clean", attackers=False)
    run = plot_run.load_run(path)
    assert plot_run.attacker_ids(run) == set()
    assert plot_run.attack_start(run) is None
    assert plot_run.ejections(run) == []
    out = tmp_path / "clean.png"
    plot_run.plot_runs([run], str(out))
    assert is_png(out)


def test_a_log_with_no_rounds_says_so(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps({"type": "header", "name": "empty", "config": {}}) + "\n")
    with pytest.raises(ValueError, match="no rounds"):
        plot_run.load_run(str(path))


def test_main_writes_next_to_the_logs_and_names_both_runs(tmp_path):
    off = fake_log(tmp_path / "fake_off.jsonl", name="fake_off")
    on = fake_log(tmp_path / "fake_on.jsonl", name="fake_on")
    out = plot_run.main([off, on])
    assert out == str(tmp_path / "fake_off_vs_fake_on.png")
    assert is_png(out)


def test_main_takes_an_explicit_output_path(tmp_path, log):
    out = tmp_path / "named.png"
    assert plot_run.main([log, "--out", str(out)]) == str(out)
    assert is_png(out)


def test_a_log_with_no_scores_says_so_instead_of_showing_an_empty_box(tmp_path):
    path = fake_log(tmp_path / "bare.jsonl", name="bare", attackers=False)
    lines = [json.loads(l) for l in open(path)]
    for line in lines:
        for w in line.get("workers", []):
            w.pop("norm_ratio", None)
    (tmp_path / "bare.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")

    fig = plot_run.plot_runs([plot_run.load_run(path)], str(tmp_path / "bare.png"))
    notes = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("no update scores" in note for note in notes)


def test_the_round_axis_only_shows_whole_numbers(tmp_path):
    path = fake_log(tmp_path / "short.jsonl", name="short", rounds=3)
    fig = plot_run.plot_runs([plot_run.load_run(path)], str(tmp_path / "short.png"))
    ticks = fig.axes[-1].get_xticks()
    assert all(float(t).is_integer() for t in ticks)
