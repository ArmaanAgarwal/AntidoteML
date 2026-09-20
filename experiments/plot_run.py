"""Plot accuracy, attack success, and worker update norms from run logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(path: str | Path) -> dict[str, Any]:
    """Load one AntidoteML JSONL run and group its records by type."""
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

    headers = [row for row in records if row.get("type") == "header"]
    if len(headers) != 1:
        raise ValueError(f"{path} must contain exactly one header record")

    summaries = [row for row in records if row.get("type") == "summary"]
    return {
        "path": path,
        "header": headers[0],
        "rounds": [row for row in records if row.get("type") == "round"],
        "summary": summaries[-1] if summaries else {},
    }


def _attacker_ids(run: dict[str, Any]) -> set[int]:
    config = run["header"].get("config") or {}
    ids = {int(worker_id) for worker_id in (config.get("attackers") or {})}
    for row in run["rounds"]:
        for worker in row.get("workers", []):
            if worker.get("role") in {"backdoor", "faulty"}:
                ids.add(int(worker["id"]))
    return ids


def _attack_start(run: dict[str, Any]) -> int | None:
    attackers = (run["header"].get("config") or {}).get("attackers") or {}
    starts = [
        int(spec["start_round"])
        for spec in attackers.values()
        if spec.get("start_round") is not None
    ]
    return min(starts) if starts else None


def _ejection_rounds(run: dict[str, Any]) -> list[int]:
    rounds = []
    for ejection in run["summary"].get("ejected", []):
        if ejection.get("round") is not None:
            rounds.append(int(ejection["round"]))
    return sorted(set(rounds))


def _default_output(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0].with_name(f"{paths[0].stem}_plot.png")
    return paths[0].parent / "comparison.png"


def plot_runs(
    log_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
) -> Path:
    """Save the required two-panel plot for one or two run logs."""
    paths = [Path(path) for path in log_paths]
    if len(paths) not in (1, 2):
        raise ValueError("plot_runs expects one or two run logs")

    runs = [load_run(path) for path in paths]
    output = Path(output_path) if output_path is not None else _default_output(paths)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, (metrics_ax, norms_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )
    styles = ["-", "--"]

    for run_index, run in enumerate(runs):
        name = str(run["header"].get("name") or run["path"].stem)
        rounds = run["rounds"]
        round_numbers = [int(row["round"]) for row in rounds]
        clean = [row.get("clean_acc") for row in rounds]
        asr = [row.get("asr") for row in rounds]
        style = styles[run_index]

        metrics_ax.plot(
            round_numbers,
            clean,
            linestyle=style,
            linewidth=2,
            label=f"{name} clean accuracy",
        )
        metrics_ax.plot(
            round_numbers,
            asr,
            linestyle=style,
            linewidth=2,
            label=f"{name} attack success",
        )

        attacker_ids = _attacker_ids(run)
        workers: dict[int, tuple[list[int], list[float]]] = {}
        for row in rounds:
            for worker in row.get("workers", []):
                ratio = worker.get("norm_ratio")
                if ratio is None:
                    continue
                worker_id = int(worker["id"])
                worker_rounds, ratios = workers.setdefault(worker_id, ([], []))
                worker_rounds.append(int(row["round"]))
                ratios.append(float(ratio))

        honest_labeled = False
        for worker_id, (worker_rounds, ratios) in sorted(workers.items()):
            is_attacker = worker_id in attacker_ids
            if is_attacker:
                label = f"{name} worker {worker_id} (attacker)"
            elif not honest_labeled:
                label = f"{name} honest workers"
                honest_labeled = True
            else:
                label = "_nolegend_"
            norms_ax.plot(
                worker_rounds,
                ratios,
                color="tab:red" if is_attacker else "0.65",
                linestyle=style,
                linewidth=2.2 if is_attacker else 0.9,
                alpha=1.0 if is_attacker else 0.65,
                label=label,
            )

        attack_start = _attack_start(run)
        if attack_start is not None:
            metrics_ax.axvline(
                attack_start,
                color="tab:orange",
                linestyle=style,
                alpha=0.8,
                label=f"{name} attack starts",
            )
            norms_ax.axvline(
                attack_start,
                color="tab:orange",
                linestyle=style,
                alpha=0.8,
            )
        for ejection_round in _ejection_rounds(run):
            metrics_ax.axvline(
                ejection_round,
                color="tab:purple",
                linestyle=style,
                alpha=0.8,
                label=f"{name} ejection",
            )
            norms_ax.axvline(
                ejection_round,
                color="tab:purple",
                linestyle=style,
                alpha=0.8,
            )

    metrics_ax.set_title("Model quality and backdoor success")
    metrics_ax.set_ylabel("Rate")
    metrics_ax.set_ylim(0, 1.05)
    metrics_ax.grid(alpha=0.25)
    metrics_ax.legend(loc="best", fontsize="small")

    norms_ax.set_title("Worker update norm ratios")
    norms_ax.set_xlabel("Round")
    norms_ax.set_ylabel("Norm / median norm")
    norms_ax.grid(alpha=0.25)
    handles, labels = norms_ax.get_legend_handles_labels()
    if handles:
        norms_ax.legend(handles, labels, loc="best", fontsize="small")

    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one or two AntidoteML JSONL run logs."
    )
    parser.add_argument("logs", nargs="+", help="JSONL run log(s)")
    parser.add_argument("-o", "--output", help="output PNG path")
    args = parser.parse_args()

    try:
        output = plot_runs(args.logs, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
