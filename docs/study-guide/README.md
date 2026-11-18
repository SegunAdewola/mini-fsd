# Study guide

<!-- toc -->
## Contents

- [Minimum track (about 6 hours total)](#minimum-track-about-6-hours-total)
- [How to use this](#how-to-use-this)
- [Learning path by week](#learning-path-by-week)
- [All resources](#all-resources)
- [Reference notes (look things up while building)](#reference-notes-look-things-up-while-building)
- [Where this repo differs from general advice](#where-this-repo-differs-from-general-advice)
- [Facts that change the design doc](#facts-that-change-the-design-doc)
<!-- tocstop -->

How to learn what Mini FSD needs, from outside material that is widely used to teach it: official
tutorials, university lectures, and talks by the people who wrote the tools. Every link was checked on
2026-11-17.


## Minimum track (about 6 hours total)

If you only do this, you'll be fine. Everything else in this guide is optional depth. You learn the rest
by building, with the reference notes (01–03) open and Claude explaining each concept as it comes up.

| Week | Do this | Time |
|---|---|---|
| 9 | ROS 2 tutorials: [Nodes](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Nodes/Understanding-ROS2-Nodes.html) and [Topics](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html), run inside `pixi shell` | 1 h |
| 9 | [tf2 Introduction](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html) + [REP 105](https://www.ros.org/reps/rep-0105.html) | 30 min |
| 9 | [MuJoCo tutorial notebook](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb), "MuJoCo basics" only | 30 min |
| 10 | Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | 40 min |
| 10 | Josh Tobin, [Sim2Real and Domain Randomization](https://www.youtube.com/watch?v=ac_W9IgKX2c) at 1.5x speed | 55 min |
| 11 | [Nav2 Navigation Concepts](https://docs.nav2.org/jazzy/getting_started/navigation_concepts/) | 30 min |
| 11 | Macenski, [On Use of Nav2 MPPI Controller](https://vimeo.com/879001391) (ROSCon 2023) | 30 min |
| 12 | [Tesla AI Day 2021, HydraNet](https://www.youtube.com/watch?v=j0z4FweCy4M&t=3227s), 53:47 to about 1:01 | 15 min |
| 12 | ONNX Runtime [quantization page](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) (static vs dynamic, QDQ) | 30 min |

Papers: just the abstract and figures of the ★ papers in [../papers.md](../papers.md), when you write
the report's related-work section (Week 13).

## How to use this

- **Learn just in time.** Do the week's "must" items right before you build that piece, not all up front.
  The must items are about 3–4 hours a week. The full resource lists add up to more than 45 hours, which
  is most of your project time, so treat everything else as optional.
- **Outside material teaches; reference notes remind.** Learn from the links below. Use the reference notes
  (01–03) to look up project-specific details while you build.
- **Papers** for the model comparison are in [../papers.md](../papers.md).

## Learning path by week

### Week 9 (catch-up): pixi env, MuJoCo robot, sim_node, teleop

Must (about 4 h):
- [ROS 2 Jazzy tutorials](https://docs.ros.org/en/jazzy/Tutorials.html), Python path only: Beginner CLI
  Tools (Nodes, Topics, Services, Parameters, Launching nodes), then Beginner Client Libraries (Creating a
  workspace, First package, Py publisher/subscriber). Run them inside `pixi shell`, not with the Ubuntu
  install steps.
- [tf2 tutorials](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Tf2-Main.html): Introduction,
  Quaternion fundamentals, Static broadcaster (Py), Broadcaster (Py). Then read
  [REP 105](https://www.ros.org/reps/rep-0105.html) (map → odom → base_link).
- [MuJoCo tutorial notebook](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb):
  "MuJoCo basics" and "Basic rendering" sections.

Optional: [Missing Semester 2026: Version Control](https://missing.csail.mit.edu/2026/version-control/),
[Learn Git Branching](https://learngitbranching.js.org/) (about 1 h).

### Week 10: data engine, real photos, eval harness, first E1 runs

Must:
- [Sim2Real and Domain Randomization](https://www.youtube.com/watch?v=ac_W9IgKX2c), Josh Tobin's guest
  lecture in Berkeley CS287 (83 min). Tobin wrote the original domain randomization paper.
- [mjSpec notebook](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb):
  building and randomizing scenes from Python.
- Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
  (40 min). Read it before the first E1 run.
- [CS231n Spring 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)
  lecture 9: Object Detection and Segmentation (73 min).
- [Ultralytics YOLO26 train mode](https://docs.ultralytics.com/modes/train/) and the
  [detection dataset format](https://docs.ultralytics.com/datasets/detect/), then run the
  [RF-DETR fine-tune cookbook](https://rfdetr.roboflow.com/latest/cookbooks/fine-tune_detection/).
- Northeastern RC: [SSH keys](https://rc.northeastern.edu/2026/10/23/ssh-instructions-for-explorer/)
  (password login stopped working Nov 6, 2026), [Running Jobs](https://rc-docs.northeastern.edu/en/latest/runningjobs/index.html)
  and [GPU job submission](https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html). Submit one
  tiny GPU job before you need a real one.
- Calibration: Shree Nayar's [First Principles of Computer Vision](https://fpcv.cs.columbia.edu/), "Camera
  Calibration" module, before you shoot the checkerboard photos.

### Week 11: E1–E3 results, Nav2 driving to room goals

Must:
- [Nav2 Navigation Concepts](https://docs.nav2.org/jazzy/getting_started/navigation_concepts/), then the
  [First-Time Robot Setup Guide](https://docs.nav2.org/jazzy/configuration_and_development/first_time_robot_setup_guide/)
  (skip its Gazebo pages).
- Steve Macenski's ROSCon talks on [MPPI](https://vimeo.com/879001391) and
  [Smac](https://vimeo.com/showcase/9954564/video/767157646), right before you tune them.
- Cyrill Stachniss, [Mobile Sensing and Robotics 1](https://www.youtube.com/playlist?list=PLgnQpQtFTOGQEn33QDVGJpiZLi-SlL7vA):
  "Occupancy Grid Maps" and "Robot Motion Planning using A*".
- CS231n lectures 8 (Attention and Transformers, covers ViT) and 12 (Self-Supervised Learning, the
  basis for DINO).
- Hugging Face [semantic segmentation guide](https://huggingface.co/docs/transformers/tasks/semantic_segmentation),
  which fine-tunes SegFormer-B0, your E2 candidate.
- Try your hallway photos in the [Depth Anything 3 demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-3),
  then run the repo's `notebooks/da3.ipynb`.
- Request DINOv3 weight access on Hugging Face early. It's gated, and approval can take time.

### Week 12: E4–E6, perception into the costmap

Must:
- Nav2 [semantic segmentation tutorial](https://docs.nav2.org/jazzy/tutorials/general_tutorials/navigation2_with_semantic_segmentation/navigation2_with_semantic_segmentation/),
  the closest existing example of camera perception feeding the costmap.
- Stachniss, "Model Predictive Control Part 1" (same playlist, 42 min), the idea under MPPI.
- [Tesla AI Day 2021](https://www.youtube.com/watch?v=j0z4FweCy4M&t=2924s), 48:44 to 1:13:12, Karpathy's
  vision section. HydraNet starts at 53:47.
- Stanford CS330 (Fall 2022) [lecture 2](https://www.youtube.com/playlist?list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI),
  Multi-Task Learning Basics: loss weighting and negative transfer, for explaining E4.
- PyTorch [Export to ONNX](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
  (`dynamo=True`), then ONNX Runtime [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
  (static vs dynamic, QDQ). Quantize in ONNX Runtime; `torch.ao.quantization` is being removed.
- Optional: Song Han's MIT 6.5940 [L5 Quantization I](https://www.youtube.com/watch?v=Qg_3N8pdK9s) and
  [L9 Knowledge Distillation](https://www.youtube.com/watch?v=JIPaWplqj_A), for the distilled depth head.

### Week 13: report and Dec 15 demo

- Google's [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), section "A
  scientific approach to improving model performance". It's useful for writing up how you compared
  models fairly.
- The "If short on time" sections of the papers marked ★ in [../papers.md](../papers.md), for related work.

## All resources

Ranked lists, at most three per topic, with the exact chapters or videos to do:

- [Robotics, sim and tooling](resources-robotics.md): ROS 2, TF2, Nav2, planning and control theory,
  MuJoCo, sim-to-real, git/pytest/mypy/pixi/Google style.
- [Computer vision and deep learning](resources-cv.md): CS231n and Michigan EECS 498, detection, SegFormer
  and DINOv3, depth, multi-task and Tesla talks, calibration and geometry, ONNX/quantization/CoreML,
  experiment practice, SLURM on Explorer.

Known stale material: Articulated Robotics and most free Robotics Back-End videos are Foxy/Humble era
(the concepts hold, the commands don't). pixi's ROS 2 tutorial uses the Humble channel. Old
`docs.nav2.org` links without `/jazzy/` return 404. The Michigan EECS 498 website is gone (the YouTube
lectures remain). The W&B courses page is gone (the notebooks live in the `wandb/edu` GitHub repo).

## Reference notes (look things up while building)

Project-specific notes written for Mini FSD. Each has a table of contents.

1. [ROS 2, TF, Nav2 and the macOS setup](01-ros-nav2.md)
2. [Simulation, data and labels](02-sim-data.md)
3. [CV models, metrics, edge deployment and tooling](03-cv-edge-tooling.md)

## Where this repo differs from general advice

- **Robot base.** Note 2 shows a wheeled robot on a `<freejoint/>`. Our plan uses a planar base (x, y, yaw
  joints with velocity actuators) driven only with unicycle-style commands. It moves like a diff-drive
  robot, with no wheel slip to tune.
- **Style config.** Note 3's example uses line length 88 and `mypy --strict`. This repo uses 80 columns
  (Google style) and the settings in `pyproject.toml`.
- **Rendering thread.** Note 2 found offscreen rendering works from a worker thread on this M1. We still
  render on the main thread in `sim_node` because it is the simplest safe rule.

## Facts that change the design doc

These came out of the research and will be logged in `docs/implementation-plan.md`:

- Depth Anything 3 Small gives relative depth only. Scale recovery from camera height is required.
- DINOv3 weights are gated under Meta's DINOv3 License (use restrictions). Note it in the report.
- EfficientViT is unmaintained, with no B0 segmentation weights for our use. Use the timm ImageNet backbone,
  or prefer SegFormer-B0 as the second E2 candidate.
- RF-DETR Nano has 30.5M parameters against 2.4M for YOLO26n, so E1 is a real latency trade.
- CoreML INT8 on an M1 saves size, not time (Apple says the speedup needs A17 Pro/M4). The Neural Engine is not
  reachable from Docker. CoreML is the first thing to cut from E6.
- Explorer's `gpu` partition: 8 h max, 1 GPU per job. Training must checkpoint and resume.
- MuJoCo 4.0.1 is current. Avoid 4.0.0 (solver bug).
