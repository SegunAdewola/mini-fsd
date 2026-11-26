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
With --latex and --columns, a booktabs table for the report is written too
(columns as metric@dataset, so one table can mix metrics). With --plot, the
same numbers are drawn as accuracy against latency (log
scale), the front as steps and the latency budget as a dashed line.
"""

import argparse
import dataclasses
import json
import math
import pathlib
from collections.abc import Mapping, Sequence
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
    p50_ms: float | None  # None for untimed runs (made elsewhere)
    p95_ms: float | None
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


def _timed(ms: float | None) -> float | None:
    # The harness logs NaN for an untimed candidate (E3a).
    return None if ms is None or math.isnan(ms) else ms


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
                    p50_ms=_timed(row["p50_ms"]),
                    p95_ms=_timed(row["p95_ms"]),
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


def with_latency(runs: Sequence[Run], clean: Sequence[Run]) -> list[Run]:
    """Runs with their latency replaced by the clean runs' (same dataset).

    Accuracy doesn't change between reruns; latency does, with heat and
    whatever else the machine is doing. So the report takes latency only
    from runs made in one cool session (a separate log), and a candidate
    without one shows none.
    """
    timed = {(r.candidate, r.dataset): r for r in clean}
    out = []
    for run in runs:
        source = timed.get((run.candidate, run.dataset))
        out.append(
            dataclasses.replace(
                run,
                p50_ms=source.p50_ms if source else None,
                p95_ms=source.p95_ms if source else None,
            )
        )
    return out


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


# Names for the report; a candidate without one keeps its harness name.
NAMES = {
    "floor-classical": "Classical baseline",
    "floor-segformer-b0": "SegFormer-B0, v9",
    "floor-segformer-b0-v10": "SegFormer-B0, v10",
    "floor-dinov2-s": "DINOv2-S frozen, v9",
    "floor-dinov2-s-v10": "DINOv2-S frozen, v10",
    "detect-yolo26n": "YOLO26n",
    "detect-rfdetr-nano": "RF-DETR Nano",
    "depth-anything-v2-small": "Depth Anything V2-S",
    "depth-da3-small": "Depth Anything 3-S",
    "depth-student": "Distilled student",
    "hydra-floor": "HydraNet-lite head",
    "hydra-detect": "HydraNet-lite head",
    "hydra-depth": "HydraNet-lite head",
}
METRIC_LABELS = {
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP@0.5",
    "ap50_door": "Door AP",
    "ap50_obstacle": "Obstacle AP",
    "ap50_room_sign": "Sign AP",
    "ap50_person": "Person AP",
    "abs_rel_global": "AbsRel",
    "delta1_global": "$\\delta_1$",
}
LATEX_MISSING = "--"


def dataset_label(dataset: str) -> str:
    """A short name for a dataset: sim:data/sim/v10/val becomes v10 val."""
    if ":" not in dataset:
        return dataset
    parts = pathlib.PurePosixPath(dataset.split(":", 1)[1]).parts
    return " ".join(parts[-2:])


def latex(
    runs: Sequence[Run],
    task: str,
    columns: Sequence[tuple[str, str]],
    latency_from: str,
    names: Mapping[str, str] = NAMES,
) -> str:
    """A booktabs tabular, one row per candidate, best first on columns[0].

    Args:
        runs: From latest().
        task: Which candidates.
        columns: (metric, dataset) per column.
        latency_from: The dataset whose p50 and p95 are shown.
        names: Display names; others keep their harness names.

    Returns:
        A tabular environment; the front (columns[0] against p50) is
        marked with a dagger.
    """
    runs = [r for r in runs if r.task == task]
    by_key = {(r.candidate, r.dataset): r for r in runs}
    candidates = sorted({r.candidate for r in runs})

    def value(name: str, metric: str, dataset: str) -> float | None:
        run = by_key.get((name, dataset))
        if run is None or metric not in run.metrics:
            return None
        v = run.metrics[metric]
        return None if math.isnan(v) else v

    first_metric, first_dataset = columns[0]
    scored, points = [], []
    for name in candidates:
        acc = value(name, first_metric, first_dataset)
        timed = by_key.get((name, latency_from))
        if acc is not None and timed is not None and timed.p50_ms is not None:
            scored.append(name)
            points.append((acc, timed.p50_ms))
    front = {scored[i] for i in pareto_front(points)}
    ranked = sorted(
        candidates,
        key=lambda n: -(value(n, first_metric, first_dataset) or -1.0),
    )
    header = ["Model"]
    header += [
        f"{METRIC_LABELS.get(m, m)} {dataset_label(d)}" for m, d in columns
    ]
    header += ["p50 ms", "p95 ms"]
    out = [
        "\\begin{tabular}{@{}l" + "r" * (len(header) - 1) + "@{}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for name in ranked:
        label = names.get(name, name) + ("$^\\dagger$" if name in front else "")
        cells = []
        for metric, dataset in columns:
            v = value(name, metric, dataset)
            cells.append(LATEX_MISSING if v is None else f"{v:.3f}")
        timed = by_key.get((name, latency_from))
        for ms in (timed.p50_ms, timed.p95_ms) if timed else (None, None):
            cells.append(LATEX_MISSING if ms is None else f"{ms:.0f}")
        out.append(" & ".join([label, *cells]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
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
            NAMES.get(x.candidate, x.candidate),
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
    ax.set_ylabel(
        f"{METRIC_LABELS.get(metric, metric)} on {dataset_label(dataset)}",
        color=INK,
        fontsize=8,
    )
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
    parser.add_argument(
        "--latex",
        type=pathlib.Path,
        help="also write a LaTeX table here, with --columns",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=[],
        metavar="METRIC@DATASET",
        help="LaTeX table columns, e.g. iou@nyuv2-val precision@nyuv2-val",
    )
    parser.add_argument(
        "--latency-log",
        type=pathlib.Path,
        help="take latency only from this log of clean, cool-session runs",
    )
    args = parser.parse_args(argv)
    runs = latest(load(args.log), args.threads)
    if args.latency_log:
        runs = with_latency(runs, latest(load(args.latency_log), args.threads))
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
    if args.latex:
        columns = [
            (spec.split("@", 1)[0], spec.split("@", 1)[1])
            for spec in args.columns or [f"{args.metric}@{args.datasets[0]}"]
        ]
        args.latex.write_text(
            latex(runs, args.task, columns, args.latency_from)
        )
        print(f"latex: {args.latex}")


if __name__ == "__main__":
    main()
