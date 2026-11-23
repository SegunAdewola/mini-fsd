# Mini FSD implementation plan

The build order. Follow it top to bottom. Each numbered step is one commit.

<!-- toc -->

## Contents

- [How to use this](#how-to-use-this)
- [Milestones](#milestones)
- [Testing ladder](#testing-ladder)
- [When surprises happen](#when-surprises-happen)
- [M0 Bootstrap](#m0-bootstrap)
- [M1 Walking skeleton](#m1-walking-skeleton)
- [M2 Data engine and real data](#m2-data-engine-and-real-data)
- [M3 Model comparison E1 to E3](#m3-model-comparison-e1-to-e3)
- [M4 Model perception in the loop, E4 and E5](#m4-model-perception-in-the-loop-e4-and-e5)
- [M5 Edge and safety](#m5-edge-and-safety)
- [M6 Report and demo](#m6-report-and-demo)
- [Open follow-ups](#open-follow-ups)
- [Hours log](#hours-log)
- [Decisions and changes log](#decisions-and-changes-log)

<!-- tocstop -->

## How to use this

- Work one step at a time, in order within a track. A step is done when its test
  is green, lint and types pass, and it is committed.
- Declare a learning mode at the start of each step: A (I write, Claude
  coaches), B (Claude writes the failing test and a skeleton, I fill the holes),
  or C (Claude writes boilerplate, I explain it back). Default for core logic is
  B; boilerplate is C.
- TDD always: write the failing test named in the step, watch it fail for the
  right reason, then implement.
- One step is one commit, using the message shown. Mention the step in the
  commit body, for example "Implements step V1".
- Tags: `[CV]` computer vision, `[R]` robotics, `[infra]` repo and tooling. Log
  hours per tag in the [hours log](#hours-log). The course asks for about 50%
  CV.
- Check off a step (`- [x]`) in the same commit that completes it.

## Milestones

| Milestone                       | Goal                                                         | Target date                       |
| ------------------------------- | ------------------------------------------------------------ | --------------------------------- |
| M0 Bootstrap                    | Mature repo: env, tooling, CI, GitHub                        | Nov 22                            |
| M1 Walking skeleton             | Every box exists in thin form, data flows end to end         | Nov 24 (progress update, halfway) |
| M2 Data engine and real data    | 10k randomized sim frames, real val/test photos, calibration | Nov 28                            |
| M3 Model comparison E1 to E3    | Detection, floor segmentation, depth picked on real val      | Dec 2                             |
| M4 Model perception in the loop | Chosen model feeds the Nav2 costmap; E4 and E5               | Dec 5                             |
| M5 Edge and safety              | E6 ONNX/INT8 latency table, speed sweep, safety monitor      | Dec 7                             |
| M6 Report and demo              | IEEE report and code (Dec 8), demo script (Dec 15)           | Dec 8 / Dec 15                    |

M1 runs two tracks side by side. The CV track (V steps) needs no ROS, so it
fills the time when the robot track (R steps) is blocked, for example during
`pixi install`.

## Testing ladder

What "tested" means at each level:

1. **Unit.** Pure logic in `src/minifsd`, no ROS, no files. `pytest`.
1. **Property.** Invariants that must hold for any input, checked with
   Hypothesis on hundreds of generated cases. Used wherever a rule is easier to
   state than a list of examples: generator invariants, round trips (pixel to
   floor to pixel), metric properties (IoU is symmetric and in [0, 1]).
1. **Node smoke.** One ROS node runs in-process for a couple of seconds and
   publishes what it should. Lives in each ROS package's `test/`.
1. **Launch integration.** The full launch comes up and topics flow at the
   expected rates (`ros2 topic hz`). Run by hand at milestone ends; scripted
   later if time allows.
1. **End to end.** A recorded sim run that reaches a typed room goal, and the
   demo script on a folder of images.

Floats are compared with `pytest.approx` or `numpy.testing.assert_allclose`,
with a tolerance that has a stated reason.

## When surprises happen

The plan is a living document. If reality diverges (an assumption breaks, a
better approach appears, a library misbehaves):

1. Stop. Do not silently deviate.
1. Discuss and decide with Claude in plan mode.
1. Update the affected steps here and add a dated entry to the
   [decisions and changes log](#decisions-and-changes-log).
1. If it changes the design, write an ADR in `docs/decisions/`.
1. Commit the doc update as its own `docs:` commit before resuming code.

## M0 Bootstrap

- [x] 0.1 `[infra]` docs: implementation plan, repo map, docs index, ADR
  template and ADRs 0003 to 0006. Test: links resolve; markdownlint clean after
  0.2.
- [x] 0.2 `[infra]` Rename the repo to `~/Downloads/mini-fsd`, create the public
  GitHub repo `SegunAdewola/mini-fsd`, push `main`. Done before any pixi
  install, because pixi environments store absolute paths. To move it later,
  clone fresh and run `pixi install` there. Test: `git remote -v` shows the
  repo; README renders on GitHub.
- [x] 0.3 `[infra]` chore: Google style toolchain (ruff, mypy, mdformat,
  markdownlint, Vale, shellcheck, shfmt, yamllint, actionlint), pixi lint
  environment and tasks, one-time markdown reflow, README, LICENSE, PR template.
  Test: `pixi run -e lint lint` clean.
- [x] 0.4 `[infra]` build: pixi env and `pixi.lock`; spike checks. Test:
  talker/listener talk; `rviz2` opens; Nav2 is installed; MuJoCo renders
  offscreen inside the env. Write ADR 0001 with measured env size and versions.
  (The `map_server` check moved to R8, which serves the real map.)
- [x] 0.5 `[infra]` build: `src/minifsd` package (hatchling, editable install
  through pixi), `tests/`, Hypothesis. Test: `pixi run test` passes a smoke test
  that imports `minifsd`.
- [x] 0.6 `[infra]` ci: GitHub Actions lint and test jobs on `macos-26`. Tests
  run in a ROS-free pixi `core` environment, so a ROS import in `src/minifsd`
  fails CI. No branch protection (solo project, direct pushes to `main`). Test:
  `actionlint` passes; the first push goes green.

## M1 Walking skeleton

Goal: every box in the architecture exists in a thin form and data flows end to
end. Big integration problems show up now, not in December.

### CV track

- [x] V1 `[CV]` feat(eval): `minifsd.eval.metrics` with `floor_iou`, `abs_rel`
  and `delta1`. Test: hand-computed cases, plus properties: IoU is in [0, 1],
  symmetric, and 1 for identical masks; `abs_rel(x, x) == 0`;
  `delta1(x, x) == 1`; both depth metrics ignore invalid (zero or non-finite)
  ground truth.
- [x] V2 `[CV]` feat(data): `minifsd.data.nyuv2` loader for the NYU Depth V2
  labeled set (images, depth, floor mask from the class named "floor", looked up
  in the file) with the official 795/654 split. Test: a tiny synthetic `.mat`
  fixture with known values; shapes, dtypes and the floor mask match.
- [x] V3 `[CV]` feat(perception): `minifsd.perception.floor_classical`, region
  growing from a bottom-center seed with an edge stop. Test: a synthetic image
  with a known floor region gives IoU above 0.95; the mask is empty or contains
  the seed (property); nothing above the horizon (property). Tune on NYUv2
  train, report floor IoU on NYUv2 test once.
- [ ] V4 `[CV]` feat(geometry): `minifsd.geometry.camera` pinhole model,
  checkerboard calibration (OpenCV) and the ground-plane mapping from pixel to
  floor metres. Test: calibration on OpenCV's sample chessboard images converges
  with reprojection error below 0.5 px; on MuJoCo-rendered checkerboards it
  recovers the known fx and fy within 1%; pixel to floor to pixel round-trips
  (property); pixels above the horizon return no point.
- [ ] V5 `[CV]` feat(perception): zero-shot Depth Anything V2 Small on NYUv2
  test, with metric scale from a least-squares fit (report both) and M1 latency.
  Needs the pixi `cv` environment (torch, transformers); discuss before adding.
  Test: output shape and dtype on one image; AbsRel below a loose bound on 20
  images.
- [ ] V6 `[CV]` feat(eval): `minifsd.eval.harness` CLI. Any candidate on any
  split gives accuracy plus p50 and p95 latency and peak memory under a thread
  cap. Test: a fake candidate with known output and a fixed sleep gives the
  expected metric and a latency within tolerance.
- [ ] V7 `[CV]` feat(data): `minifsd.data.datagen` renders RGB, floor mask and
  depth from the sim (needs R3), split by floor-plan seed. Test: no seed in two
  splits (property); mask and depth line up with the RGB frame.
- [ ] V8 `[CV]` feat(demo): `minifsd.demo` CLI, folder of images in, overlay
  images out (classical floor baseline for now). Test: two input images give two
  output files of the same size.

### Robot track

- [x] R1 `[R]` feat(world): `minifsd.world.floor_plan.generate_floor_plan`, a
  seeded hallway with numbered rooms. Test (properties over seeds): room numbers
  unique; door gaps open; hallway closed elsewhere; spawn clear of walls; same
  seed gives the same layout; `rooms_per_side=0` raises `ValueError`.
- [x] R2 `[R]` feat(world): `minifsd.world.mjcf.build_mjcf`, a planar base (ADR
  0002\) with a front camera. Test: compiles; named joints, actuators and camera
  exist; the camera looks along +x at 0.5 m; one sign plate per room.
- [x] R3 `[R]` feat(world): `minifsd.world.sim.Sim` with unicycle drive, command
  watchdog, clamping and RGB plus depth rendering. Test: drives 0.5 m in 1 s;
  turns 1 rad in 1 s; stops within 0.5 s of the last command; clamps at 3 m/s;
  walls block it; depth to the end wall is right within 5 cm; `body_to_world`
  keeps speed (property).
- [x] R4 `[R]` feat(geometry): `minifsd.geometry.frames`, quaternions, the
  optical frame rotation and camera intrinsics from field of view. Test: optical
  axes map to base axes; fy for 240 px at 50 degrees; quaternions stay unit
  length and yaw round-trips (property).
- [ ] R5 `[R]` feat(sim): ROS package `mini_fsd_sim` with `sim_node` publishing
  `/clock`, `/odom`, `/tf`, `/tf_static`, camera, depth and camera info;
  subscribes `/cmd_vel`. Test: in-process smoke test receives odom and an image,
  and odom x grows after a forward command.
- [ ] R6 `[R]` feat(bringup): `mini_fsd_bringup` launch file, RViz config,
  `pixi run sim` and `pixi run teleop`. Test (by hand): RViz shows the camera;
  teleop drives; walls stop the robot; closing teleop stops it. Record the Week
  9 video.
- [ ] R7 `[R]` feat(world): `minifsd.world.occupancy`, layout to occupancy grid
  (PGM plus YAML for `map_server`). Test: wall cells occupied, door cells free
  (property).
- [ ] R8 `[R]` feat(bringup): Nav2 on the generated map with ground-truth pose
  (ADR 0004), Smac 2D planner and MPPI controller. Test (by hand): a goal
  clicked in RViz reaches the target.
- [ ] R9 `[R]` feat(mission): `minifsd.mission.goals.parse_goal` ("go to room
  103" to a pose in front of that door) and `mission_node` sending
  `NavigateToPose`. Test: every room in a layout resolves (property); unknown
  rooms raise.
- [ ] R10 `[R]` feat(perception): `perception_node` stub, depth image to
  PointCloud2 in the robot frame, into the local costmap. Reuses V4's
  projection. Test: a pixel's point projects back to that pixel (property);
  obstacles appear in the costmap in RViz.

## M2 Data engine and real data

Steps are written when M1 is done. Scope: domain randomization (textures,
lighting, camera height 0.3 to 1.6 m, distractors, blur, noise), sign text, COCO
boxes, about 10k frames split by seed; about 150 real photos labeled with SAM 2
help (floor 1 = val, rest = test); Pixel 8a calibration.

## M3 Model comparison E1 to E3

Steps written at the end of M2. E1 detection (YOLO26n vs RF-DETR Nano), E2 floor
segmentation (classical vs SegFormer-B0 vs frozen DINOv3 plus head), E3 depth
(Depth Anything V2 Small vs DA3 Small vs distilled head). Trained on Explorer,
picked on real val.

## M4 Model perception in the loop, E4 and E5

Steps written at the end of M3. The chosen model replaces the R10 stub. E4
HydraNet-lite vs separate models. E5 sim-only vs sim plus DR vs sim plus DR plus
50 real images.

## M5 Edge and safety

Steps written at the end of M4. E6 FP32 vs FP16 vs INT8 (ONNX Runtime CPU) under
the budget profiles; speed vs latency sweep; sign reading; safety monitor.

## M6 Report and demo

IEEE report and code on Dec 8. Dec 15 demo script (ONNX Runtime CPU, weights
bundled, offline), rehearsal on held-out photos.

## Open follow-ups

Not decided yet. Each has the step where it must be settled.

- **Real photos depend on campus access** (M2). If campus slips past about Dec
  1, shoot the real set in another indoor building and lean on NYUv2 for the
  public benchmark. Say so in the report.

- **Depth Anything 3 Small gives relative depth only** (V5, M3). Metric scale
  must come from known camera height and the floor plane.

- **DINOv3 weights are gated** under Meta's DINOv3 License (M3). Request access
  early; note the license in the report.

- **EfficientViT is unmaintained** and has no usable B0 segmentation weights
  (M3). Use the timm ImageNet backbone or drop it for SegFormer-B0.

- **CoreML INT8 on an M1 saves size, not time** (M5). The Neural Engine is not
  reachable from Docker. First thing to cut from E6.

- **Explorer limits** (M3): `gpu` partition 8 h max, 1 GPU per job; SSH keys
  required since Nov 6, 2026. Training must checkpoint and resume.

- **YOLO26 is AGPL-3.0** (M3). Decide whether that affects the repo license (ADR
  0006\) or how ultralytics is used.

- **Torch on the M1** (V5). Torch and transformers go in a separate pixi
  environment so the ROS env stays small. Confirm disk space first.

- **The campus floors are open plan, not corridors** (V7, M2). Floors 2, 14 and
  15 are open areas around a central core (8 elevators, 2 stairwells), with desk
  clusters and meeting rooms on the edge. R1 only makes corridors. Add a second
  layout type to the generator: open floor, central core, furniture clusters.
  Plans live in `private/floor-plans/` (never committed).

- **Glass walls** (V7, M3). Most rooms on the campus floors have glass walls and
  doors. Monocular depth tends to see through glass. Render some walls as glass
  in the generator, shoot glass on purpose for the real set, and report it as a
  known failure case.

- **Angled and curved walls** (digital twin, after R8). `Wall` is axis-aligned.
  A twin of a campus floor needs walls with an angle; R2 and R7 change by a few
  lines when that happens.

- **Floor-plan scale** (M2). Assumed elevator door 1.1 m; confirm one real
  measurement on campus and rescale if needed.

- **Render tests on a Linux runner** (M2). GitHub's macOS runners have no GPU,
  so CI deselects the `gl` tests. Add a `linux-64` platform and run them there
  with software rendering (EGL or OSMesa) before the data engine depends on
  rendering.

## Hours log

Add a row at the end of each session.

| Date       | CV hours | R hours | Infra hours | Steps done                    |
| ---------- | -------- | ------- | ----------- | ----------------------------- |
| 2026-11-17 | 0        | 0       | 2           | Study guide, repo conventions |

## Decisions and changes log

- **2026-11-17** pixi instead of conda for the environment (RoboStack is
  conda-only; pixi gives a lockfile and per-project env). Study guide built from
  outside sources, with project reference notes.
- **2026-11-21** Walking skeleton first, with a CV track and a robot track. All
  ROS-free logic in `src/minifsd` (ADR 0003). Google style toolchain (ADR 0005).
  Public repo under Apache-2.0 (ADR 0006). CV work uses NYUv2 and synthetic data
  until real photos are possible.
- **2026-11-22** Spike passed; ADR 0001 records pixi + RoboStack with measured
  numbers (6.4 GB env, 2 ms offscreen render). The `map_server` spike check
  moved from 0.4 to R8, where the real map is served anyway.
- **2026-11-22** CI tests in a ROS-free `core` env; branch protection dropped to
  save time. Real photos: one building, floors 2, 14 and 15, not the 2 to 3
  extra buildings in the proposal. Floors 14 and 15 share a layout, so
  validation is floor 15 and test is floors 14 and 2 (floor 2 is the unseen
  layout). Claims are limited to unseen floors of a known building; NYUv2 test
  covers unseen buildings. R1 renamed from "building" to "floor plan", since it
  makes one floor and "floor" alone already means the floor mask.
- **2026-11-22** V3 evaluates NYUv2 on the Eigen crop (rows 45 to 470, cols 41
  to 600); without it the seed lands on the white border and IoU is 0.01. Only
  about 40% of NYUv2 images have floor at the bottom centre, so the baseline
  reports IoU on all images and on that subset. Settings tuned on train
  (tolerance 2, Canny 80/200, train IoU 0.278).
- **2026-11-23** V3 result, NYUv2 test (654 images, run once): floor IoU 0.236
  overall, 0.440 on the 266 images with floor at the bottom centre; 1.5 ms p50
  on the M1. The worst images are all cases where the seed missed the floor.
- **2026-11-23** R3 render tests fail on CI (`CGLError: invalid pixel format`):
  the macOS runners are VMs with no GPU. Render tests carry a `gl` marker that
  CI deselects; they run locally before each commit. Linux render CI is a
  follow-up.
