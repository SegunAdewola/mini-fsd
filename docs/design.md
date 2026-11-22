# Design: Mini FSD, camera-only self-driving for an indoor robot (CS5330 final project)

Project name: **Mini FSD** (a small-scale, indoor version of Tesla FSD)

Author: Oluwasegun Adewola (solo). Course: CS5330. Dates: Nov 10 – Dec 8, 2026
(code + report), live demo Dec 15.

## Context and why

Oluwasegun presented Tesla FSD in class (camera-only perception, HydraNet,
bird's-eye-view occupancy, data engine, simulation). The final project scales
that idea down to an indoor robot in simulation. The goals:

1. Get a strong grade on a CV final project. The professor requires about 50% of
   the time to go to computer vision, and the system must run on **unseen real
   images the professor provides on Dec 15**.
1. Build a portfolio project aimed at robotics SDE roles (Amazon Robotics SDE
   Early Career 2027 and similar). That posting asks for solid software
   engineering (data structures, OOP, distributed and fault-tolerant systems,
   CI/CD, AWS, optimization math such as LP and nonlinear optimization) plus
   preferred experience in perception, CV, path planning, and navigation.
1. Learn the real industry stack, ROS 2 especially. Oluwasegun explicitly wants
   ROS 2 even if it costs setup time.

Oluwasegun is also doing thesis research (KVSpec: specifying and fault-testing
KV-cache movement in LLM servers), so time is limited. The design should borrow
that research habit (explicit invariants plus fault injection) without creating
extra work.

All three demo moments matter equally: the high-speed dodge, sim-to-real
perception, and "go to room X" language-to-motion. The project should feel like
one polished end-to-end system, not one deep piece with a thin wrapper.

## What it does

Input: a typed instruction such as "go to room XYZ" and a simulated building.
Output: the robot plans a route, drives there using only its forward camera for
obstacle sensing, dodges obstacles that appear in its path, confirms arrival by
reading the door sign, and logs latency, speed, and safety events.

Separately, for the Dec 15 demo, the perception network runs on a folder of real
indoor photos. For each photo it outputs a drivable-floor mask, obstacle and
door boxes, a depth map, and any room number it reads, all drawn as an overlay.

## Constraints

- Hardware: Apple M1, 8 GB RAM, macOS 26. Training happens on Northeastern
  Explorer (NVIDIA GPUs) and Colab.
- **The M1 is the "edge computer."** Perception plus planning must run in real
  time on it under an explicit budget (target: perception on 2 CPU threads, or
  on the CoreML/ANE path, plus a fixed memory cap). This stands in for a Jetson
  Orin Nano-class robot computer.
- Solo, about 2.5 mini-projects of work, five weeks, which includes Thanksgiving
  week (Nov 26). **Oluwasegun has 10–15 hours a week** because of the thesis.
  Only the Must tier is committed; Should items are done only if Must is on
  track at the Nov 24 checkpoint.
- No LiDAR phone, so real depth ground truth comes from NYU Depth V2 only. Our
  own photos get floor masks, boxes, and room-number labels, but no depth
  labels.
- The professor has not said what the Dec 15 images will be or how they will be
  judged. So the proposal defines what the system handles, and the camera-height
  randomization covers 0.3–1.6 m (robot height up to a handheld phone), so
  either kind of photo works.
- About 50% of the time goes to CV: data engine, models, sim-to-real evaluation,
  compression, OCR.
- Plain, human writing in all reports and docs. No AI-sounding phrasing.

## Decisions made (Oluwasegun's)

- Simulation, not a physical robot (agreed with the professor).
- Solo project.
- **ROS 2 is the core middleware from day one**, accepting the setup cost.
- Balanced end-to-end polish across the three demo moments.

## Architecture (my calls unless noted)

```text
mission_node      "go to room XYZ" -> room lookup -> goal pose (Nav2 NavigateToPose action)
Nav2              map_server (generated building map), Smac planner (global), MPPI controller (local)
perception_node   /camera/color -> HydraNet-lite -> floor mask + boxes + depth
                  -> PointCloud2 of obstacle points in robot frame -> Nav2 local costmap
sign_reader_node  at the goal, crop the door sign from detections and run OCR to confirm the room number
safety_monitor    watches invariants (min distance vs. speed, perception staleness) and can force a stop
sim_node          MuJoCo: diff-drive robot, procedural building, moving obstacles;
                  publishes /camera/color, /camera/depth (ground truth only, for eval), /odom, /tf, /clock;
                  subscribes /cmd_vel
fault_injector    optional node that drops or delays camera frames and adds latency spikes
```

- **ROS 2 Jazzy via RoboStack (conda, native osx-arm64).** Nav2 packages exist
  on the robostack-jazzy channel for osx-arm64. Fallback: Docker `ros:jazzy`
  arm64 (Ubuntu 24.04), with the edge budget enforced via `--cpus` and
  `--memory`. Last-resort fallback: a UTM Ubuntu VM. Decide by the end of Week 9
  based on a spike.
- **MuJoCo (pinned version)** for the world. Prefer a small custom Python
  `sim_node` (MuJoCo Python bindings + rclpy) over `mujoco_ros2_control`,
  because it may not be packaged for osx-arm64 and a diff-drive base only needs
  cmd_vel in and odom out. Note: MuJoCo camera axes differ from the ROS optical
  frame, so handle that in TF.
- **Localization is out of scope.** Use the simulator's true pose for odom/TF
  and a known generated map. The report states this assumption.
- **Visualization:** Foxglove (via foxglove_bridge) or RViz2, plus Rerun for
  perception debugging. The builder picks whichever runs cleanly on the M1.
- **Nav2 stock planners first.** MPPI is a sampling-based model predictive
  controller, which covers the "optimization" story. Stretch: write a custom C++
  Nav2 controller or costmap plugin.

## The CV part (about 50%)

1. **Data engine.** A procedural building generator (hallways, rooms, numbered
   door signs, furniture and boxes as obstacles) with domain randomization: wall
   and floor textures, lighting, camera height/pitch jitter, distractors, motion
   blur, noise. MuJoCo renders RGB, depth, and segmentation, so labels are free.
   Target: about 10k labeled frames at 320×240 or 384×288.

   - Split **by building seed** (for example 80/10/10 across seeds), not by
     frame, so test buildings are never seen in training.
   - Export: COCO JSON for boxes (classes: obstacle, person-like mover, door,
     room sign), PNG masks for floor, 16-bit depth PNG.

1. **Real data.**

   - Real test set: about 150 phone photos of Northeastern hallways and rooms,
     taken at two heights (about 0.5 m robot height and about 1.5 m handheld).
     Labels: floor masks made with SAM 2 assistance, boxes for doors, signs, and
     obstacles, and room-number text. Kept strictly as test data, except for one
     ablation that fine-tunes on a small labeled subset of about 50.
   - Public real benchmark: NYU Depth V2 labeled set (1,449 images, real depth
     plus a floor class) for depth error and floor IoU.

1. **Models: a measured comparison, not an assumed winner** (Oluwasegun's call).
   Published benchmarks disagree by hardware (RF-DETR looks better on NVIDIA
   GPUs, YOLO26n is 4–5x faster on CPU), so every choice is made on our own data
   and our own M1 edge budget.

   **Protocol (the same for every candidate):**

   - One shared eval harness: same splits, input size, epoch/time budget, and
     seeds. It reports accuracy plus p50/p95 latency and peak memory on the M1
     budget.
   - Real **validation** set (~30 of our photos plus an NYUv2 val subset) for
     picking models. The real **test** set is used once, at the end, for the
     reported numbers.
   - Decision rule: the most accurate model that fits the latency budget (whole
     perception stack ≤ 50 ms, so 20 Hz). Results shown as accuracy-vs-latency
     plots (Pareto front).

   **Experiments:**

   - E1 Detection (obstacle / mover / door / room sign): YOLO26n vs. RF-DETR
     Nano, both fine-tuned on sim.
   - E2 Floor segmentation: classical baseline (region-growing from the
     bottom-center plus edge cues) vs. an efficient CNN/transformer
     (EfficientViT-B0 or SegFormer-B0, ImageNet-pretrained) vs. a frozen DINOv3
     ViT-S/16 with a light head. Hypothesis: the foundation features cross the
     sim-to-real gap better, which is an open question in the literature.
   - E3 Depth: Depth Anything 3 Small (zero-shot) vs. Depth Anything V2 Small
     (zero-shot) vs. our small distilled depth head, evaluated on NYUv2.
   - E4 Architecture, the Tesla question: separate specialist models vs. one
     shared-backbone **HydraNet-lite** (backbone = the E2 winner that fits the
     budget, with heads for floor, detection, and depth distilled from the E3
     teacher). Does sharing cost accuracy, and how much latency does it save?
   - E5 Training data, on the chosen model: sim-only vs. sim + domain
     randomization vs. sim + DR + ~50 real labeled images.
   - E6 Compression, on the chosen model: FP32 vs. FP16 vs. INT8 (ONNX Runtime
     CPU and CoreML).

   Cost control: candidates use existing training code (Ultralytics, the
   `rfdetr` package, Hugging Face), so most of the work is configs, and GPU jobs
   run unattended on Explorer. The harness is the real engineering. If behind,
   keep 2 candidates per experiment and drop E6's CoreML path.

1. **Sign reading.** An off-the-shelf OCR model (for example PaddleOCR) on the
   room-sign crops. Not trained by us.

1. **Bird's-eye costmap.** Floor mask plus depth gives obstacle points in the
   robot frame, published as PointCloud2 to Nav2's local costmap (the
   small-scale version of Tesla's BEV/occupancy).

## Evaluation

- Perception (real test set and NYUv2): floor IoU, detection mAP@0.5 for
  doors/signs/obstacles, depth AbsRel and δ\<1.25, room-number OCR accuracy.
- Model comparison tables and accuracy-vs-latency plots for E1–E6 (selection on
  real val, final numbers on real test).
- Edge: latency (p50, p95), FPS, and peak memory for each model variant under
  the fixed budget.
- Navigation (sim, held-out building seeds, N goals each): success rate,
  collision rate, time to goal, safety-monitor stops.
- **Headline plot:** maximum safe speed against perception latency. Sweep the
  speed cap (for example 0.5–3 m/s) with each model variant, with obstacles that
  pop out at fixed distances. This connects "fast model" to "fast robot."
- Fault injection: success and collision rates with dropped frames, added
  latency, and stale perception, with and without the safety monitor.

Success criteria for the proposal (the builder can tune these after the first
numbers):

- On the real photo test set, the chosen perception model (sim + DR) clearly
  beats the classical baseline on floor IoU (target ≥ 0.80).
- Perception runs at ≥ 20 Hz on the edge budget after INT8.
- At least 90% navigation success on held-out buildings at a meaningful speed
  (target ≥ 1.5 m/s scaled sim units), with fewer collisions than the heavy
  model at the same speed.
- Dec 15: on unseen real indoor photos (hallways, classrooms, offices, normal
  indoor lighting, camera anywhere from robot height to handheld), it produces
  sensible floor, obstacle, and depth overlays and reads clear room-number
  signs.

## Scope tiers

- **Must:** ROS 2 + MuJoCo sim + Nav2 driving to typed room goals; data engine;
  the shared eval harness; E1–E5 with at least 2 candidates each; real val +
  test sets (~150 photos total); perception feeding the Nav2 costmap; INT8/ONNX
  export with a latency table under the budget (this is the core of the edge
  story and costs little); Dec 15 image demo script.
- **Should:** speed-vs-latency sweep; sign-reading confirmation at the goal
  (off-the-shelf OCR); safety monitor.
- Heavy baseline kept cheap: YOLO26n fine-tuned on sim (a few hours of training)
  plus Depth Anything 3 Small used as-is, with no DA3 training.
- **Stretch:** fault-injection study; custom C++ Nav2 plugin; GitHub Actions CI
  with ROS launch tests; S3 for datasets and checkpoints (AWS signal); a
  free-form language instruction parser with an LLM.

## Engineering practices (portfolio signals)

Git and GitHub from day one; ROS 2 packages (`ament_python`, plus `ament_cmake`
for any C++); typed Python and pytest unit tests for the generator, label
export, costmap projection, and the mission parser; a launch file that brings up
the whole stack with one command; W&B for training runs; pinned environments
(conda env file / Dockerfile); a README with a GIF and the results table.

## Weekly plan sketch (feeds the proposal table)

- **Week 8 (Nov 3–9):** background research on Tesla FSD; class presentation.
  Deliverable: the Tesla presentation.
- **Week 9 (Nov 10–16):** proposal and pitch. ROS 2 Jazzy + Nav2 environment
  spike on the M1 (RoboStack, fallback Docker). MuJoCo diff-drive robot and the
  first procedural building, publishing camera, odom, and TF, driven by teleop.
  Deliverable: environment file, RViz/Foxglove screen recording of the robot
  driving, a one-paragraph decision note on RoboStack vs. Docker.
- **Week 10 (Nov 17–23):** finish the generator with domain randomization and
  label export, then render about 10k frames split by building seed. Shoot and
  label about 150 real photos (SAM 2-assisted) and split them into val and test
  by building. Write the shared eval harness and the classical baseline, then
  launch the first E1 runs (YOLO26n, RF-DETR Nano) on Explorer. Deliverable:
  dataset with a counts table per split, labeled real val/test sets, the harness
  CLI printing classical-baseline floor IoU and latency.
- **Week 11 (Nov 24–30, progress update Nov 24, Thanksgiving):** finish E1–E3 on
  Explorer and pick the winners on real val. Connect Nav2 to the generated map
  and get the mission node driving to typed room goals. Deliverable: E1–E3
  tables and accuracy-vs-latency plots, a video of an end-to-end "go to room X"
  run with ground-truth obstacles.
- **Week 12 (Dec 1–7):** E4 and E5 results; perception_node feeding the costmap
  from the chosen model; E6 (ONNX/INT8) and the final latency table under the
  budget. If on track: the speed-vs-latency sweep, the sign reader, and the
  safety monitor. Deliverable: latency table, a demo video with camera-only
  obstacle avoidance, and (if done) the speed-vs-collision plot.
- **Week 13 (Dec 8–14):** submit the code and IEEE report on Dec 8. Then the Dec
  15 demo script (folder of images in, overlays out), a rehearsal, and fixes.
  Deliverable: report PDF, repo, demo script tested on held-out photos.

## Calls I made after the reviewer check (please check these)

- Pretrained backbones (ImageNet/COCO) are allowed, since the course guide
  itself uses "fine-tune YOLOv8n" as an example. "Sim-only" in the ablation
  means sim-only task labels on a pretrained backbone.
- Unlabeled real frames for depth distillation come from NYU raw video and from
  buildings that are never used in the real test set, so the test set stays
  clean.
- Small training runs start in Week 10 so Thanksgiving week isn't the only
  training window.
- The Dec 15 demo is a standalone script (ONNX Runtime CPU, no ROS, weights
  bundled, works offline) run live on the M1: folder of images in, overlays out.
  The dodge and "go to room X" moments are shown as recorded videos, plus a live
  sim run only if the stack is stable.
- Real test photos are taken at two heights (about 0.5 m and about 1.5 m) to
  match the 0.3–1.6 m sim randomization.

## Left to the builder

The exact backbone, head designs, and loss weights; image resolution; Foxglove
vs. RViz; the obstacle class list beyond the four above; the robot's physical
dimensions; exact success thresholds once first numbers exist; the repo layout.

## Immediate next steps after approval

1. Save this design to
   `~/Downloads/final-project/docs/2026-11-10-edge-nav-robot-design.md`.
1. Write the proposal PDF due today at 3:15 PM PT: a one-page proposal plus the
   Week 8–13 table, in plain language, with the dataset links (NYU Depth V2) and
   access confirmed.
1. Write short pitch notes for the five-minute talk (problem, data and approach,
   Dec 15 success, biggest risk = ROS 2 on M1 and the sim-to-real gap).
1. Save a feedback memory: no AI-sounding language in chat or reports.
1. Then run writing-plans for the Week 9 implementation work.

## Verification

- Proposal: check against every rubric row (problem, data with
  link/size/format/split/access, methods + baseline, metrics + success + demo
  images, every week with a task and a deliverable, scope, risks).
- Week 9 spike: `ros2 launch` brings up sim + Nav2 + RViz on the M1, and teleop
  moves the robot with the camera image visible.
- Throughout: pytest passes; the eval scripts print the results tables from
  saved checkpoints.

## Risks and fallbacks

1. ROS 2/Nav2 on M1 8 GB is unstable → Docker arm64 → UTM VM → as a last resort,
   keep the same node boundaries in plain Python and drop Nav2 for a
   hand-written A\* + DWA.
1. The sim-to-real gap is too large on real photos → heavier domain
   randomization, pseudo-labels from SAM 2/DA3 on unlabeled real frames, the
   50-image real fine-tune.
1. Running out of time → cut from the Stretch tier first, then the speed sweep.
   The Must tier still makes a complete project.

## Updates after the Nov 10 discussion

- **Compute budget profiles (Must).** The robot software runs in a Docker
  container with hard caps; the simulator sits outside the caps (it is the
  world, not the robot). Profiles: 6 cores/8 GB (about Orin NX 16 GB), 4 cores/4
  GB with perception on 2 threads (about Orin Nano 8 GB, shared), and 2 cores/2
  GB (Raspberry Pi 5 class). M1 cores are faster than Jetson ARM cores, and the
  report says so. Optional: a second measurement on an AWS Graviton t4g
  instance, if the Learner Lab allows it.
- **Digital twin.** The three Vancouver campus floors are rebuilt in MuJoCo from
  the Robin floor plans, scaled using standard door widths (about 0.9 m), with
  estimated furniture. Procedural buildings remain the main training source.
  Ablation: procedural only vs. procedural + twin. Don't publish the floor plans
  or twin geometry in a public repo without the campus's OK.
- **Real photos.** Pixel 8a, main 1x lens only. About 40 per floor (120) plus
  about 40 from 2–3 other buildings. Split: floor 1 = validation (and the
  50-photo fine-tune pool); floors 2–3 + other buildings = test. The camera is
  calibrated with a printed checkerboard (OpenCV) so real photos can be lifted
  into a bird's-eye map. Plus a few slow knee-height walking videos (unlabeled
  frames for distillation, and a demo clip).
- **Cameras (Should).** Front plus left/right at about ±65°, fused geometrically
  into one bird's-eye costmap. Experiment: 1 vs. 3 cameras, coverage against
  compute. A learned BEV (Lift-Splat style, trained on sim ground truth) is
  Stretch.
- **Negative obstacles (Should).** (1) Drive only on positively seen floor; (2)
  depth + RANSAC ground plane, flagging points below it (Stretch); (3) a
  simulated downward cliff sensor watched by the safety monitor. Pits and stair
  drops added to the generator.
- **Recovery and backing up.** Use Nav2's behavior-tree recovery (wait, clear
  costmap, back up, spin). Backing up relies on the costmap's memory of what was
  seen on the way in, plus a simulated rear proximity sensor. No rear camera
  unless the multi-camera work is done early. Metric: recoveries per run.
- **Fleet (extra scope, in the proposal only if time permits).** 3–5 robots; a
  central fleet manager assigns tasks by linear-programming assignment and
  prevents robots from blocking each other in hallways. Targets the Amazon
  posting's distributed-systems and optimization lines.
- **Stereo depth (added to E3).** The MuJoCo robot gets an optional left/right
  front camera pair (about 6–12 cm apart; the builder picks it) rendered with
  exact depth. Classical stereo (OpenCV SGBM) is compared against the
  single-camera models on sim depth accuracy (AbsRel, δ\<1.25, with blank-wall
  hallways reported separately), latency under the three hardware limits, and
  the resulting navigation success. If stereo wins, the simulated robot uses it.
  The Dec 15 photo demo always uses the best single-camera model, since the demo
  images are single photos and our real photos are not stereo pairs. Stretch:
  one learned stereo network. Single-camera scale fix to report: recover metric
  scale from the known camera height and the detected floor plane.
