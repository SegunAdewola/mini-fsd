r"""The experiment harness: any candidate on any dataset, scored the same way.

    pixi run -e cv python -m minifsd.eval.harness \\
        --candidate depth-anything-v2-small --dataset nyuv2-test --threads 2

Every E1 to E6 number goes through here, so all candidates share the same
data, metrics, latency measurement and compute budget (design, "Protocol").
It reports accuracy, p50 and p95 latency and the process's peak memory, and
appends one JSON line per run to runs/harness/results.jsonl with the git
commit, so each table in the report traces back to an exact run.

The compute budget is a thread cap (design: perception on 2 threads). It is
set before any model library loads, because OpenMP and BLAS read their
thread counts once, at import.

Peak memory is the process's peak resident size. Samples stream from disk one
at a time, so it is mostly the model, not the dataset; run one candidate per
process (one command per run) for clean numbers.
"""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import re
import resource
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal

import numpy as np

from minifsd.eval import depth as depth_eval
from minifsd.eval import depth_align
from minifsd.eval import floor as floor_eval

Task = Literal["floor", "depth"]
RESULTS = pathlib.Path("runs/harness/results.jsonl")
# Read once by OpenMP, Apple Accelerate, OpenBLAS and MKL when they load.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
NYU = pathlib.Path("data/nyuv2")
FIT_IMAGES = 100  # train images for depth's global scale and shift (V5)


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A model under test: RGB in, a floor mask or relative disparity out."""

    name: str
    task: Task
    predict: Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Result:
    """One candidate on one dataset."""

    candidate: str
    task: Task
    dataset: str
    n_images: int
    metrics: dict[str, float]
    p50_ms: float
    p95_ms: float
    peak_rss_mb: float
    threads: int


def set_thread_cap(n: int) -> None:
    """Caps every library's thread pool at n threads.

    Call before importing torch or onnxruntime; their pools are sized at
    import. Libraries already loaded (OpenCV, torch) are set directly too.

    Args:
        n: Threads, at least 1.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)
    import cv2

    # OpenCV's macOS backend (GCD) honours only 0 ("no threading", so 1
    # thread); any other value means "all cores". Run it single-threaded so
    # it never exceeds the cap; it only does light work (preprocessing, the
    # classical baseline), so this costs little.
    cv2.setNumThreads(n if n > 1 and _opencv_pool_is_settable() else 0)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)


def _opencv_pool_is_settable() -> bool:
    import cv2

    return (
        re.search(r"Parallel framework:\s+GCD", cv2.getBuildInformation())
        is None
    )


def peak_rss_mb() -> float:
    """The process's peak resident memory so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def evaluate(
    candidate: Candidate,
    samples: Iterable[Any],
    dataset: str,
    threads: int,
    fit: Iterable[Any] | None = None,
) -> Result:
    """Runs a candidate over samples and scores it for its task.

    Args:
        candidate: The model.
        samples: Floor samples (rgb, floor, labeled) or depth samples (rgb,
            depth), streamed.
        dataset: The dataset's name, for the record.
        threads: The thread cap in force, for the record.
        fit: Depth only: samples for the global scale and shift (from a
            training split, never the one being scored).

    Returns:
        The result. Floor metrics: iou, iou_bottom_floor. Depth metrics:
        abs_rel and delta1, each with the per-image (oracle) and the global
        fit.

    Raises:
        ValueError: For a depth candidate without fit samples.
    """
    metrics: dict[str, float]
    if candidate.task == "floor":
        f = floor_eval.evaluate_floor(samples, candidate.predict)
        n, p50, p95 = f.n_images, f.p50_ms, f.p95_ms
        metrics = {"iou": f.iou, "iou_bottom_floor": f.iou_bottom_floor}
    else:
        if fit is None:
            raise ValueError("a depth candidate needs fit samples")
        global_fit = depth_align.fit_many(
            (candidate.predict(s.rgb), s.depth) for s in fit
        )
        d = depth_eval.evaluate_depth(samples, candidate.predict, global_fit)
        n, p50, p95 = d.n_images, d.p50_ms, d.p95_ms
        metrics = {
            "abs_rel_oracle": d.per_image_fit.abs_rel,
            "delta1_oracle": d.per_image_fit.delta1,
            "abs_rel_global": d.global_fit.abs_rel,
            "delta1_global": d.global_fit.delta1,
            "global_scale": global_fit.scale,
            "global_shift": global_fit.shift,
        }
    return Result(
        candidate=candidate.name,
        task=candidate.task,
        dataset=dataset,
        n_images=n,
        metrics=metrics,
        p50_ms=p50,
        p95_ms=p95,
        peak_rss_mb=peak_rss_mb(),
        threads=threads,
    )


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() + ("-dirty" if dirty else "")


def append_result(
    result: Result, path: pathlib.Path = RESULTS, **extra: object
) -> None:
    """Appends the result as one JSON line, with the commit and the time.

    Args:
        result: The result.
        path: The log; created if missing.
        **extra: More settings to record, e.g. device="mps".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        **dataclasses.asdict(result),
        **extra,
        "git_commit": _git_commit(),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _floor_classical(device: str, threads: int) -> Candidate:
    del device, threads  # OpenCV only, capped by set_thread_cap
    from minifsd.perception import floor_classical

    return Candidate("floor-classical", "floor", floor_classical.floor_mask)


def _depth_anything(device: str, threads: int) -> Candidate:
    import torch

    from minifsd.perception import depth_anything

    torch.set_num_threads(threads)
    model = depth_anything.DepthAnything(
        pathlib.Path("data/models/depth-anything-v2-small"), device=device
    )
    return Candidate("depth-anything-v2-small", "depth", model.predict)


# Name -> factory. Factories import their libraries, so the harness loads
# only the one being tested. M3 adds the trained models here.
CANDIDATES: dict[str, Callable[[str, int], Candidate]] = {
    "floor-classical": _floor_classical,
    "depth-anything-v2-small": _depth_anything,
}


def make_candidate(name: str, device: str, threads: int) -> Candidate:
    """Builds a registered candidate.

    Raises:
        ValueError: For an unknown name.
    """
    if name not in CANDIDATES:
        raise ValueError(
            f"unknown candidate {name!r}; known: {list(CANDIDATES)}"
        )
    return CANDIDATES[name](device, threads)


def load_dataset(name: str, limit: int | None = None) -> Iterator[Any]:
    """Streams a dataset's samples.

    Names: "nyuv2-train", "nyuv2-test" (Eigen crop), "nyuv2-fit" (the fixed
    sample of FIT_IMAGES train images used for depth's global fit), and
    "sim:<folder>" for a data engine split such as sim:data/sim/v7/val.

    Raises:
        ValueError: For an unknown name.
    """
    if name.startswith("sim:"):
        from minifsd.data import datagen

        frames = datagen.read_frames(pathlib.Path(name[4:]))
        for i, frame in enumerate(frames):
            if limit is not None and i >= limit:
                return
            yield frame
        return
    if name not in ("nyuv2-train", "nyuv2-test", "nyuv2-fit"):
        raise ValueError(f"unknown dataset {name!r}")
    from minifsd.data import nyuv2

    splits = nyuv2.load_split(NYU / "splits.mat")
    if name == "nyuv2-fit":
        rng = np.random.default_rng(0)  # the same sample as V5
        indices = sorted(
            int(i)
            for i in rng.choice(splits["train"], FIT_IMAGES, replace=False)
        )
    else:
        indices = splits[name.removeprefix("nyuv2-")]
    with nyuv2.NyuV2(NYU / "nyu_depth_v2_labeled.mat") as data:
        for i in indices[:limit]:
            yield nyuv2.crop(data[i])


def main(argv: Sequence[str] | None = None) -> None:
    """Scores one candidate on one dataset and logs the result."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--limit", type=int, help="score only the first N")
    parser.add_argument(
        "--fit",
        default="nyuv2-fit",
        help="depth only: dataset for the global scale and shift",
    )
    parser.add_argument("--out", type=pathlib.Path, default=RESULTS)
    args = parser.parse_args(argv)

    set_thread_cap(args.threads)  # before any model library loads
    candidate = make_candidate(args.candidate, args.device, args.threads)
    candidate.predict(np.zeros((240, 320, 3), np.uint8))  # warm-up
    fit = load_dataset(args.fit) if candidate.task == "depth" else None
    result = evaluate(
        candidate,
        load_dataset(args.dataset, args.limit),
        args.dataset,
        args.threads,
        fit,
    )
    append_result(result, args.out, device=args.device, fit=args.fit)

    print(f"{result.candidate} on {result.dataset} ({result.n_images} images)")
    for key, value in result.metrics.items():
        print(f"  {key:18s} {value:.4g}")
    print(
        f"  latency p50/p95    {result.p50_ms:.1f} / {result.p95_ms:.1f} ms "
        f"({args.device}, {args.threads} threads)"
    )
    print(f"  peak memory        {result.peak_rss_mb:.0f} MB")
    print(f"  logged to          {args.out}")


if __name__ == "__main__":
    main()
