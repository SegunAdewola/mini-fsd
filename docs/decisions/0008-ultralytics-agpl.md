# ADR 0008: Train YOLO26n with Ultralytics, keep the repo Apache-2.0

- Status: Accepted (revisit at T5 if YOLO26n wins E1)
- Date: 2026-11-25
- Plan step: E1a

## Context

E1 compares two detectors: YOLO26n (Ultralytics) and RF-DETR Nano (Roboflow,
Apache-2.0). The `ultralytics` package and its pretrained weights are AGPL-3.0,
and Ultralytics treats models trained with them, and their exports, as covered
by that license too. ADR 0006 licensed the repo under Apache-2.0 and left this
for M3.

The package also brings practical trouble: it requires the PyPI `opencv-python`,
which would sit next to conda's OpenCV in our pixi environments, the clash
`pyproject.toml` already warns about.

## Decision

- `ultralytics` (pinned, 8.4.214) is installed on Colab only, in the notebook.
  Only `minifsd.train.yolo` imports it, inside its training function, so the
  rest of the package never loads it and the local environments don't have it.
- Training exports the best model to ONNX. The harness scores that file with
  ONNX Runtime and our own pre- and postprocessing, which is also how the demo
  runs models (design, M6). No Ultralytics code runs on the M1.
- Weights are never committed (as for every model).
- Our code stays Apache-2.0. If YOLO26n wins E1 and goes into the demo, the demo
  is distributed under the AGPL's terms. The source is public already, so that
  costs little, and the README will say which parts are covered. That choice is
  confirmed at T5.

## Consequences

- E1 can start today, on the detector the design named first.
- The comparison stays fair: both detectors are scored through ONNX Runtime on
  the M1 with the same harness.
- If RF-DETR wins, none of this matters for the shipped system.
- Exporting to ONNX does not escape the AGPL; it only keeps Ultralytics' code
  out of our runtime and our environments.

## Alternatives considered

- **Relicense the whole repo AGPL-3.0 now.** Simple, but it restricts reuse
  before we know whether YOLO26n is even kept.
- **Skip YOLO26n; compare RF-DETR Nano against another Apache detector.** Clean
  licensing, but YOLO26n is the design's first candidate and the usual baseline,
  so leaving it out weakens E1.
- **Install `ultralytics` in the local `cv` environment.** Lets the harness call
  it directly, but two OpenCV builds in one environment is a known source of
  crashes.
