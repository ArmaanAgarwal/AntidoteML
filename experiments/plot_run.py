"""Turn one or two run logs into the demo picture. Owner: Person 4.

    python experiments/plot_run.py runs/backdoor_off.jsonl runs/backdoor_on.jsonl

Top panel: clean accuracy and attack success per round, with a line where the
attack starts and a line where a worker is ejected. Bottom panel: how big each
worker's update was next to the group, with the attacker picked out.

Reads only the log, so it never needs the model, the data or a rerun.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # saving a file, never opening a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Two hues carry identity and nothing else does. Checked for colour blind
# separation rather than picked by eye.
CLEAN = "#2a78d6"  # clean accuracy
ATTACK = "#eb6834"  # attack success, and the attacker's own line below
HONEST = "#898781"  # every honest worker, one muted band
EJECT_LINE = "#0ca30c"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

NAN = float("nan")


@dataclass
class Run:
    name: str
    config: dict
    rounds: list
    summary: dict
    path: str


def load_run(path):
    """Read a runs/<name>.jsonl into a Run."""
    header, rounds, summary = {}, [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "round":
                rounds.append(record)
            elif kind == "summary":
                summary = record
    if not rounds:
        raise ValueError(f"{path} has no rounds in it, so there is nothing to plot")
    name = header.get("name") or os.path.splitext(os.path.basename(path))[0]
    return Run(name=name, config=header.get("config") or {}, rounds=rounds,
               summary=summary, path=path)


def round_numbers(run):
    return [r["round"] for r in run.rounds]


def number(value):
    """A float for the plot. Anything missing becomes a gap in the line."""
    if value is None:
        return NAN
    try:
        value = float(value)
    except (TypeError, ValueError):
        return NAN
    return NAN if math.isnan(value) else value


def metric(run, key):
    return [number(r.get(key)) for r in run.rounds]


def worker_ids(run):
    ids = {w["id"] for r in run.rounds for w in r.get("workers", [])}
    return sorted(ids)


def attacker_ids(run):
    """Ground truth, straight from the config. Only the picture sees this, never
    the detector."""
    ids = set()
    for key in (run.config.get("attackers") or {}):
        ids.add(int(key))
    for r in run.rounds:
        for w in r.get("workers", []):
            if w.get("role") not in (None, "honest"):
                ids.add(w["id"])
    return ids


def attack_start(run):
    """The round the attack switches on, from the events or from the config."""
    starts = [
        r["round"]
        for r in run.rounds
        for e in r.get("events", [])
        if e.get("type") == "attack_started"
    ]
    for spec in (run.config.get("attackers") or {}).values():
        if isinstance(spec, dict) and spec.get("start_round") is not None:
            starts.append(int(spec["start_round"]))
    return min(starts) if starts else None


def ejections(run):
    """(worker id, round) pairs, from the summary, the events, or the statuses."""
    found = {}
    for row in run.summary.get("ejected") or []:
        if row.get("round") is not None:
            found.setdefault(row["worker_id"], row["round"])
    for r in run.rounds:
        for e in r.get("events", []):
            if "eject" in str(e.get("type", "")) and e.get("worker_id") is not None:
                found.setdefault(e["worker_id"], r["round"])
        for w in r.get("workers", []):
            if w.get("status") == "ejected":
                found.setdefault(w["id"], r["round"])
    return sorted(found.items())


def norm_ratios(run):
    """worker id -> one value per round, with a gap wherever it did not take part."""
    ids = worker_ids(run)
    series = {wid: [NAN] * len(run.rounds) for wid in ids}
    for i, r in enumerate(run.rounds):
        for w in r.get("workers", []):
            if w.get("participated") is False:
                continue
            series[w["id"]][i] = number(w.get("norm_ratio"))
    return series


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    # Rounds are whole numbers, so 1.25 on the axis would be nonsense.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def mark_events(ax, run, labels):
    """The two moments that explain the whole plot.

    The lines are drawn on both panels so the eye can read straight down, but
    only the top panel carries the words, or the text lands on the data.
    """
    marks = []
    start = attack_start(run)
    if start is not None:
        marks.append((start, "attack starts", INK_MUTED))
    for wid, rnd in ejections(run):
        marks.append((rnd, f"worker {wid} ejected", EJECT_LINE))
    for at, text, color in marks:
        ax.axvline(at, color=color, linestyle="--", linewidth=1.2, zorder=1)
        if labels:
            ax.annotate(
                text,
                xy=(at, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -3),
                textcoords="offset points",
                color=INK_SOFT,
                fontsize=8,
                rotation=90,
                va="top",
                # A hairline of surface behind the words keeps them readable
                # where they cross a line.
                bbox=dict(boxstyle="square,pad=0.12", facecolor=SURFACE,
                          edgecolor="none", alpha=0.85),
            )


def last_point(x, series):
    for i in range(len(series) - 1, -1, -1):
        if not math.isnan(series[i]):
            return x[i], series[i]
    return None


def label_end(ax, at, text, dy=0):
    """Name the line at its own end, so the reader never hunts in a legend."""
    if at is None:
        return
    ax.annotate(
        text,
        xy=at,
        xytext=(5, dy),
        textcoords="offset points",
        color=INK_SOFT,
        fontsize=8,
        va="center",
    )


def put_legend(ax, ncols):
    """Above the axes, where it can never sit on top of a line."""
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncols=ncols,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SOFT,
        handlelength=1.6,
        borderaxespad=0.3,
    )


def plot_scores(ax, run):
    x = round_numbers(run)
    clean = metric(run, "clean_acc")
    asr = metric(run, "asr")
    dots = "o" if len(x) <= 15 else None

    ax.plot(x, clean, color=CLEAN, linewidth=2, marker=dots, markersize=4,
            label="clean accuracy")
    ax.plot(x, asr, color=ATTACK, linewidth=2, marker=dots, markersize=4,
            label="attack success")
    style_axes(ax)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("share of test images", color=INK_SOFT, fontsize=9)
    mark_events(ax, run, labels=True)

    # Both lines can finish at the same height, which prints one label on top
    # of the other. Nudge them apart when that happens.
    clean_end, asr_end = last_point(x, clean), last_point(x, asr)
    apart = 0
    if clean_end and asr_end and abs(clean_end[1] - asr_end[1]) < 0.08:
        apart = 6
    label_end(ax, clean_end, "clean", dy=apart)
    label_end(ax, asr_end, "attack", dy=-apart)
    put_legend(ax, 2)


def plot_norms(ax, run):
    x = round_numbers(run)
    series = norm_ratios(run)
    bad = attacker_ids(run)

    honest_drawn = False
    for wid, values in series.items():
        if wid in bad:
            continue
        ax.plot(x, values, color=HONEST, linewidth=1.2, alpha=0.75,
                label="honest workers" if not honest_drawn else "_nolegend_")
        honest_drawn = True
    for wid in sorted(bad):
        ax.plot(x, series.get(wid, []), color=ATTACK, linewidth=2.4,
                label=f"worker {wid}, the attacker")

    style_axes(ax)
    ax.axhline(1.0, color=AXIS, linewidth=1, zorder=1)
    ax.set_xlabel("round", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("update size vs the group", color=INK_SOFT, fontsize=9)

    finite = [v for values in series.values() for v in values if not math.isnan(v)]
    if not finite:
        # A defense off run before the detector exists, or a log written by an
        # older coordinator. Say so rather than showing an empty box.
        ax.text(
            0.5, 0.5, "no update scores in this log",
            transform=ax.transAxes, ha="center", va="center",
            color=INK_MUTED, fontsize=9,
        )
    top = max(finite) if finite else 1.0
    if top >= 4:
        # One scaled update dwarfs the honest band. A log axis keeps both
        # readable instead of flattening nine workers onto the floor.
        ax.set_yscale("log")
        ax.set_ylabel("update size vs the group (log)", color=INK_SOFT, fontsize=9)
    mark_events(ax, run, labels=False)
    if honest_drawn or bad:
        put_legend(ax, 2)


def plot_runs(runs, out_path):
    """Two panels per run, side by side when there are two runs."""
    fig, axes = plt.subplots(
        2, len(runs), figsize=(6.4 * len(runs), 7.2), sharex="col", squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for column, run in enumerate(runs):
        plot_scores(axes[0][column], run)
        plot_norms(axes[1][column], run)
        axes[0][column].set_title(
            f"{run.name}   defense {'on' if run.config.get('defense') else 'off'}",
            color=INK,
            fontsize=11,
            pad=26,  # room for the legend, which sits above the axes
        )
    fig.suptitle(
        "AntidoteML: what the coordinator saw, round by round",
        color=INK,
        fontsize=13,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    return fig


def default_out(runs):
    folder = os.path.dirname(runs[0].path) or "."
    return os.path.join(folder, "_vs_".join(r.name for r in runs) + ".png")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot one or two AntidoteML run logs")
    parser.add_argument("logs", nargs="+", help="runs/<name>.jsonl, one or two of them")
    parser.add_argument("--out", help="where to save the png")
    args = parser.parse_args(argv)
    if len(args.logs) > 2:
        parser.error("one or two logs, not more")

    runs = [load_run(path) for path in args.logs]
    out_path = args.out or default_out(runs)
    fig = plot_runs(runs, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
