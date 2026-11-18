# Study guide

Everything you need to know to build Mini FSD, from the big picture down to the commands. Facts and
versions were checked on 2026-11-17 from official docs, model cards and papers, and the MuJoCo code in
part 2 was run on this M1.

Each topic has the same six parts: why it matters for the project, big picture, in the weeds, gotchas,
a 10–20 minute "try it" exercise, and links to read. Read the big picture before the week you need it.
Do the "try it" when you start the work. Come back to "in the weeds" while you build.

## Parts

1. [ROS 2, TF, Nav2 and the macOS setup](01-ros-nav2.md): nodes, topics, QoS, launch, lifecycle, TF2
   frames, message types, Nav2 (costmaps, Smac, MPPI, behavior trees), RoboStack + pixi, RViz2, Foxglove.
2. [Simulation, data and labels](02-sim-data.md): MuJoCo, procedural generation and domain randomization,
   the sim-to-real gap, datasets and label formats (COCO, NYUv2, SAM 2), camera calibration, the classical
   floor baseline.
3. [CV models, metrics, edge deployment and tooling](03-cv-edge-tooling.md): YOLO26n, RF-DETR, SegFormer,
   EfficientViT, DINOv3, Depth Anything, HydraNet-lite, metrics and latency measurement, ONNX/INT8/CoreML,
   OCR, Explorer + W&B, git/pixi/ruff/mypy/pytest/pre-commit and Google style.

## Learning map by week

| Week | You are building | Read first |
|---|---|---|
| 9 (catch-up) | pixi env, MuJoCo robot, sim_node, teleop in RViz | 1.1 ROS core, 1.2 TF, 1.3 messages, 1.5 pixi, 1.6 RViz; 2.1 MuJoCo; 3.9 tooling |
| 10 | data engine, real photos, eval harness, first E1 runs | 2.2 domain randomization, 2.4 datasets, 2.5 calibration, 2.6 baseline; 3.1 detection, 3.5 metrics, 3.8 Explorer |
| 11 | E1–E3 results, Nav2 driving to room goals | 3.2 segmentation, 3.3 depth; 1.4 Nav2; 2.3 sim-to-real |
| 12 | E4–E6, perception into the costmap | 3.4 HydraNet-lite, 3.6 edge deployment, 3.7 OCR; 1.4 Nav2 (PointCloud2 costmap) |
| 13 | report, Dec 15 image demo | 3.5 metrics (reporting), 3.6 ONNX Runtime CPU |

## Where this repo differs from the guide's general advice

The guide explains each tool in general terms. Where this project made a specific choice, the repo wins:

- **Robot base.** Part 2 shows a wheeled robot on a `<freejoint/>`. Our plan uses a planar base (x, y, yaw
  joints with velocity actuators) driven only with unicycle-style commands. It moves like a diff-drive
  robot, with no wheel slip to tune. The caster gotcha in part 2 is still worth knowing.
- **Style config.** Part 3's example uses line length 88 and `mypy --strict`. This repo uses 80 columns
  (Google style) and the settings in `pyproject.toml`.
- **Rendering thread.** Part 2 found offscreen rendering works from a worker thread on this M1. We still
  render on the main thread in `sim_node` because it is the simplest safe rule.

## Facts that change the design doc

These came out of the research and are logged in `docs/implementation-plan.md`:

- Depth Anything 3 Small gives relative depth only. Scale recovery from camera height is required.
- DINOv3 weights are gated under Meta's DINOv3 License (use restrictions). Note it in the report.
- EfficientViT is unmaintained, with no B0 segmentation weights for our use. Use the timm ImageNet backbone,
  or prefer SegFormer-B0 as the second E2 candidate.
- RF-DETR Nano has 30.5M parameters against 2.4M for YOLO26n, so E1 is a real latency trade.
- CoreML INT8 on an M1 saves size, not time (Apple says the speedup needs A17 Pro/M4). The Neural Engine is not
  reachable from Docker. CoreML is the first thing to cut from E6.
- Explorer's `gpu` partition: 8 h max, 1 GPU per job. Training must checkpoint and resume.
- MuJoCo 4.0.1 is current. Avoid 4.0.0 (solver bug).
