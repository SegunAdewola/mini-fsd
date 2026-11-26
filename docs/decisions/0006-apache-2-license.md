# ADR 0006: License the repo under Apache-2.0

- Status: Accepted (M3 revisit done: see ADR 0008)
- Date: 2026-11-21
- Plan step: 0.2

## Context

The repo is public from the start, as a portfolio piece. A public repo with no
license means nobody may legally reuse it. Some planned dependencies have their
own terms: Ultralytics YOLO26 is AGPL-3.0, DINOv3 weights are under Meta's
DINOv3 License, and RF-DETR, Depth Anything 3 Small and SegFormer code are
Apache-2.0 or similar.

## Decision

License our own code under Apache-2.0. It is permissive, includes an explicit
patent grant, and is what Google uses for most of its open source. Third-party
code and weights are installed as dependencies, never copied into the repo, and
keep their own licenses.

## Consequences

- Anyone can read and reuse the code with attribution.
- Model weights are not committed, which also avoids redistributing gated or
  AGPL weights.
- **Revisit at M3.** If the final system imports Ultralytics, the AGPL may apply
  to the combined program. Options then: relicense our code under AGPL-3.0
  (easy, since it's a solo project), keep YOLO26 only in the comparison scripts,
  or export to ONNX and run without Ultralytics at inference time.

## Alternatives considered

- **MIT.** Also permissive and simpler, but no explicit patent grant.
- **AGPL-3.0 now.** Matches Ultralytics, but restricts reuse before we know
  whether YOLO26 wins E1.
