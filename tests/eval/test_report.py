"""Tests for minifsd.eval.report: results tables and Pareto fronts (T4)."""

import json
import pathlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.eval import report


def _row(
    candidate: str,
    dataset: str,
    iou: float,
    p50: float,
    time: str = "2026-11-25T10:00:00",
    threads: int = 2,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "task": "floor",
        "dataset": dataset,
        "n_images": 10,
        "metrics": {"iou": iou},
        "p50_ms": p50,
        "p95_ms": p50 * 2,
        "peak_rss_mb": 400.0,
        "threads": threads,
        "device": "cpu",
        "time": time,
    }


def _log(tmp_path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


FIXTURE = [
    _row("classical", "nyuv2-val", 0.33, 3.0),
    _row("segformer", "nyuv2-val", 0.30, 150.0, "2026-11-25T09:00:00"),
    _row("segformer", "nyuv2-val", 0.37, 134.0),  # a later rerun wins
    _row("segformer", "sim:v9", 0.99, 130.0),
    _row("dino", "nyuv2-val", 0.47, 91.0),
    _row("dino", "sim:v9", 0.96, 76.0),
    _row("dino", "nyuv2-val", 0.10, 9.0, threads=8),  # another budget
]


def test_the_latest_run_of_a_candidate_on_a_dataset_wins(
    tmp_path: pathlib.Path,
) -> None:
    rows = report.latest(report.load(_log(tmp_path, FIXTURE)), threads=2)
    seg = [r for r in rows if r.candidate == "segformer"]
    assert {(r.dataset, r.metrics["iou"]) for r in seg} == {
        ("nyuv2-val", 0.37),
        ("sim:v9", 0.99),
    }
    assert all(r.threads == 2 for r in rows)


def test_table_has_a_row_per_candidate_and_a_column_per_dataset(
    tmp_path: pathlib.Path,
) -> None:
    rows = report.latest(report.load(_log(tmp_path, FIXTURE)), threads=2)
    table = report.table(
        rows, "floor", "iou", ["nyuv2-val", "sim:v9"], latency_from="nyuv2-val"
    )
    assert [line.candidate for line in table] == [
        "dino",
        "segformer",
        "classical",
    ]
    dino = table[0]
    assert dino.scores == {"nyuv2-val": 0.47, "sim:v9": 0.96}
    assert dino.p50_ms == 91.0 and dino.p95_ms == 182.0
    # A candidate not scored on a dataset gets no number, not a zero.
    assert table[2].scores == {"nyuv2-val": 0.33, "sim:v9": None}


def test_markdown_table_lists_every_candidate_and_marks_the_front(
    tmp_path: pathlib.Path,
) -> None:
    rows = report.latest(report.load(_log(tmp_path, FIXTURE)), threads=2)
    table = report.table(
        rows, "floor", "iou", ["nyuv2-val", "sim:v9"], latency_from="nyuv2-val"
    )
    text = report.markdown(table, "iou")
    lines = text.splitlines()
    assert lines[0].startswith("| Candidate")
    assert "nyuv2-val" in lines[0] and "sim:v9" in lines[0]
    body = lines[2:]
    assert len(body) == 3
    assert "0.470" in body[0] and "91" in body[0]
    assert "—" in body[2]  # classical has no sim:v9 score
    # dino and classical are on the front; segformer is beaten by dino.
    assert [("*" in line.split("|")[1]) for line in body] == [True, False, True]


POINTS = st.lists(
    st.tuples(
        st.floats(0, 1, allow_nan=False), st.floats(0.1, 1000, allow_nan=False)
    ),
    min_size=1,
    max_size=12,
)


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] >= b[0] and a[1] <= b[1] and a != b


@given(POINTS)
def test_nothing_on_the_front_is_beaten_and_everything_off_it_is(
    points: list[tuple[float, float]],
) -> None:
    front = report.pareto_front(points)
    assert front  # a non-empty set always has a best trade-off
    for i in front:
        assert not any(_dominates(p, points[i]) for p in points)
    for i in set(range(len(points))) - set(front):
        assert any(_dominates(points[j], points[i]) for j in front)


def test_an_unknown_metric_is_an_error(tmp_path: pathlib.Path) -> None:
    rows = report.latest(report.load(_log(tmp_path, FIXTURE)), threads=2)
    with pytest.raises(KeyError):
        report.table(rows, "floor", "abs_rel", ["nyuv2-val"], "nyuv2-val")


def test_front_line_runs_from_fast_to_accurate() -> None:
    points = [(0.33, 3.0), (0.37, 134.0), (0.47, 91.0), (0.20, 50.0)]
    # Only classical (0.33, 3 ms) and dino (0.47, 91 ms) are unbeaten.
    assert report.front_line(points) == [(3.0, 0.33), (91.0, 0.47)]


def test_plot_writes_an_image_with_the_budget_line(
    tmp_path: pathlib.Path,
) -> None:
    rows = report.latest(report.load(_log(tmp_path, FIXTURE)), threads=2)
    lines = report.table(
        rows, "floor", "iou", ["nyuv2-val", "sim:v9"], latency_from="nyuv2-val"
    )
    figure = report.plot(lines, "iou", "nyuv2-val", budget_ms=50.0)
    labels = {t.get_text() for t in figure.axes[0].texts}
    assert {"dino", "segformer", "classical"} <= labels
    assert any("50 ms" in t for t in labels)
    path = tmp_path / "e2.png"
    figure.savefig(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
