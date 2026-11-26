"""Results tables and Pareto fronts from the harness log, for the report (T4).

    pixi run python -m minifsd.eval.report --task floor --metric iou
        --datasets nyuv2-val sim:data/sim/v9/val --latency-from nyuv2-val

(one command, wrapped here to fit)

The harness appends a line per run to runs/harness/results.jsonl. Reruns
happen, so for each candidate, dataset and thread budget only the latest
line counts. The table has a row per candidate, a column per dataset, and
latency from one chosen dataset (image size changes preprocessing time, so
latencies from different datasets don't mix).

The design's rule picks the most accurate model inside the latency budget.
The Pareto front shows the choices worth making: a candidate is on it when
no other is at least as accurate and at least as fast, and better in one.
Accuracy for the front is the first dataset's score.
With --plot, the same numbers are drawn as accuracy against latency (log
scale), the front as steps and the latency budget as a dashed line.
"""

import argparse
import dataclasses
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from matplotlib import figure as mpl_figure
from matplotlib import ticker

from minifsd.eval import harness

MISSING = "—"
BUDGET_MS = 50.0  # the design's budget for the whole perception stack
# From the reference palette (dataviz skill): one series, so one hue; text in
# ink colours, never the series colour.
FRONT_COLOUR = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


@dataclasses.dataclass(frozen=True)
class Run:
    """One harness log line."""

    candidate: str
    task: str
    dataset: str
    metrics: dict[str, float]
    p50_ms: float
    p95_ms: float
    threads: int
    time: str


@dataclasses.dataclass(frozen=True)
class Line:
    """One candidate's row in a table."""

    candidate: str
    scores: dict[str, float | None]  # dataset -> metric, None if not run
    p50_ms: float | None
    p95_ms: float | None
    on_front: bool


def load(path: pathlib.Path) -> list[Run]:
    """Reads every line of a harness log."""
    runs = []
    with path.open() as f:
        for text in f:
            row: dict[str, Any] = json.loads(text)
            runs.append(
                Run(
                    candidate=row["candidate"],
                    task=row["task"],
                    dataset=row["dataset"],
                    metrics=row["metrics"],
                    p50_ms=row["p50_ms"],
                    p95_ms=row["p95_ms"],
                    threads=row["threads"],
                    time=row["time"],
                )
            )
    return runs


def latest(runs: Sequence[Run], threads: int) -> list[Run]:
    """The newest run per candidate and dataset, at one thread budget."""
    newest: dict[tuple[str, str], Run] = {}
    for run in runs:
        if run.threads != threads:
            continue
        key = (run.candidate, run.dataset)
        if key not in newest or run.time >= newest[key].time:
            newest[key] = run
    return list(newest.values())


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the (accuracy, latency) points that nothing beats.

    A point is beaten when another is at least as accurate and at least as
    fast, and not the same point.
    """

    def beats(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return a[0] >= b[0] and a[1] <= b[1] and a != b

    return [
        i for i, p in enumerate(points) if not any(beats(q, p) for q in points)
    ]


def front_line(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """The front as (latency, accuracy) points, fastest first, to draw."""
    return sorted((points[i][1], points[i][0]) for i in pareto_front(points))


def table(
    runs: Sequence[Run],
    task: str,
    metric: str,
    datasets: Sequence[str],
    latency_from: str,
) -> list[Line]:
    """One line per candidate of the task, best first on datasets[0].

    Raises:
        KeyError: If a run lacks the metric.
    """
    runs = [r for r in runs if r.task == task]
    by_key = {(r.candidate, r.dataset): r for r in runs}
    candidates = sorted({r.candidate for r in runs})
    lines = []
    for name in candidates:
        scores = {
            d: by_key[(name, d)].metrics[metric]
            if (name, d) in by_key
            else None
            for d in datasets
        }
        timed = by_key.get((name, latency_from))
        lines.append(
            Line(
                candidate=name,
                scores=scores,
                p50_ms=timed.p50_ms if timed else None,
                p95_ms=timed.p95_ms if timed else None,
                on_front=False,
            )
        )
    first = datasets[0]
    names, points = [], []
    for line in lines:  # only candidates with both numbers can be ranked
        score, p50 = line.scores[first], line.p50_ms
        if score is not None and p50 is not None:
            names.append(line.candidate)
            points.append((score, p50))
    front = {names[i] for i in pareto_front(points)}
    lines = [
        dataclasses.replace(x, on_front=x.candidate in front) for x in lines
    ]
    return sorted(lines, key=lambda x: -(x.scores[first] or -1.0))


def markdown(lines: Sequence[Line], metric: str) -> str:
    """The table as Markdown; front candidates are marked with *."""
    if not lines:
        return ""
    datasets = list(lines[0].scores)
    header = ["Candidate", *(f"{d} {metric}" for d in datasets)]
    header += ["p50 ms", "p95 ms"]
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for line in lines:
        name = line.candidate + (" *" if line.on_front else "")
        cells = [
            MISSING if v is None else f"{v:.3f}" for v in line.scores.values()
        ]
        cells += [
            MISSING if v is None else f"{v:.0f}"
            for v in (line.p50_ms, line.p95_ms)
        ]
        out.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(out) + "\n"


def plot(
    lines: Sequence[Line],
    metric: str,
    dataset: str,
    budget_ms: float = BUDGET_MS,
) -> mpl_figure.Figure:
    """Accuracy against p50 latency, front joined, budget marked.

    Front candidates are filled and joined; beaten ones are hollow grey, so
    the two read apart in greyscale print too. Every point is named.
    """
    shown = [x for x in lines if x.scores.get(dataset) is not None and x.p50_ms]
    fig = mpl_figure.Figure(figsize=(4.8, 3.2), dpi=200, layout="constrained")
    ax = fig.add_subplot()
    points = [
        (float(x.scores[dataset] or 0.0), float(x.p50_ms or 0.0)) for x in shown
    ]
    if points:
        xs, ys = zip(*front_line(points), strict=True)
        # Steps, not a slope: there is no model between two front points, so
        # the best on offer at latency x is the last front point at or below x.
        ax.plot(
            xs,
            ys,
            color=FRONT_COLOUR,
            linewidth=2,
            zorder=2,
            drawstyle="steps-post",
        )
    for x, (acc, ms) in zip(shown, points, strict=True):
        ax.scatter(
            [ms],
            [acc],
            s=64,
            zorder=3,
            facecolor=FRONT_COLOUR if x.on_front else "white",
            edgecolor=FRONT_COLOUR if x.on_front else INK_SECONDARY,
            linewidth=1.5,
        )
        ax.annotate(
            x.candidate,
            (ms, acc),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    ax.axvline(budget_ms, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.text(
        budget_ms,
        0.98,
        f"{budget_ms:.0f} ms budget ",  # left of the line, clear of points
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.set_xscale("log")
    if points:  # room on the right for the slowest point's name
        fastest = min(ms for _, ms in points)
        slowest = max(ms for _, ms in points)
        ax.set_xlim(fastest / 1.5, max(slowest, budget_ms) * 3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("p50 latency, ms (log scale)", color=INK, fontsize=8)
    ax.set_ylabel(f"{metric} on {dataset}", color=INK, fontsize=8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    """Prints one experiment's table, front marked with *."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--latency-from", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log", type=pathlib.Path, default=harness.RESULTS)
    parser.add_argument(
        "--plot", type=pathlib.Path, help="also save the plot here (png/svg)"
    )
    parser.add_argument("--budget-ms", type=float, default=BUDGET_MS)
    args = parser.parse_args(argv)
    runs = latest(load(args.log), args.threads)
    lines = table(
        runs, args.task, args.metric, args.datasets, args.latency_from
    )
    print(markdown(lines, args.metric), end="")
    print(
        f"\n* on the Pareto front: {args.datasets[0]} {args.metric} against "
        f"p50 latency on {args.latency_from}, {args.threads} threads."
    )
    if args.plot:
        fig = plot(lines, args.metric, args.datasets[0], args.budget_ms)
        fig.savefig(args.plot)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
