# Repository layout

Where things live and why. Folders marked *(planned)* appear when the step that
creates them lands; see the [implementation plan](implementation-plan.md).

```text
mini-fsd/
├── src/minifsd/                 # All ROS-free logic (ADR 0003). Plain Python.
│   ├── world/                   # The simulated world
│   │   ├── building.py          # Seeded hallway + numbered rooms (R1)
│   │   ├── mjcf.py              # Layout -> MuJoCo XML, planar robot (R2)
│   │   ├── sim.py               # MuJoCo wrapper: drive, watchdog, render (R3)
│   │   ├── clutter.py           # Boxes on the floor as obstacles (V7)
│   │   ├── signs.py             # Room numbers drawn on sign plates (D7)
│   │   └── occupancy.py         # Layout -> Nav2 occupancy map (R7)
│   ├── geometry/
│   │   ├── frames.py            # Quaternions, optical frame, intrinsics (R4)
│   │   └── camera.py            # Pinhole, calibration, floor mapping (V4)
│   ├── perception/
│   │   ├── floor_classical.py   # Classical floor baseline (V3)
│   │   └── depth_zero_shot.py   # Depth Anything wrapper (V5)
│   ├── eval/
│   │   ├── metrics.py           # IoU, AbsRel, delta1 (V1)
│   │   ├── detection.py         # mAP at IoU 0.5 (T1)
│   │   └── harness.py           # Accuracy + latency CLI for E1-E6 (V6)
│   ├── data/
│   │   ├── nyuv2.py             # NYU Depth V2 loader (V2)
│   │   ├── datagen.py           # Sim frames + labels, split by seed (V7, D1-D9)
│   │   ├── camera_effects.py    # Blur, noise, exposure, JPEG on renders (D5)
│   │   ├── boxes.py             # Object boxes from renders, COCO files (D8)
│   │   └── coco_people.py       # COCO person subset for detection (D10)
│   ├── train/                   # Training on Colab (ADR 0007)
│   │   └── pack.py              # Datasets zipped for Google Drive (T2)
│   ├── mission/
│   │   └── goals.py             # "go to room 103" -> goal pose (R9)
│   └── demo/                    # Dec 15: folder of images -> overlays (V8)
├── tests/                       # pytest + Hypothesis, mirrors src/minifsd
├── notebooks/
│   └── train_colab.ipynb        # Thin Colab runner: Drive, install, train (T2)
├── ros2_ws/src/                 # ROS 2 packages: thin shells over minifsd
│   ├── mini_fsd_sim/            # sim_node: clock, odom, TF, camera (R5)
│   ├── mini_fsd_bringup/        # Launch files, RViz config, Nav2 params (R6, R8)
│   ├── mini_fsd_mission/        # mission_node (R9)  (planned)
│   └── mini_fsd_perception/     # perception_node (R10)  (planned)
├── tools/                       # One-off scripts (spike checks, downloads)
├── docs/
│   ├── README.md                # Index of all docs
│   ├── design.md                # Approved design
│   ├── implementation-plan.md   # Build order and current step
│   ├── repo.md                  # This file
│   ├── papers.md                # Reading list for the model comparison
│   ├── decisions/               # ADRs: why each design choice was made
│   ├── study-guide/             # Learning path + project reference notes
│   └── proposal/                # Nov 10 proposal and pitch notes
├── .github/                     # CI workflow and PR template
├── pixi.toml / pixi.lock        # Environment (ADR 0001); lock is generated
├── pyproject.toml               # minifsd package + ruff/mypy/pytest config
├── CLAUDE.md, .claude/          # Local agent config (gitignored, private)
├── data/                        # Datasets and renders (gitignored)
├── papers/                      # Paper PDFs (gitignored)
└── private/                     # Campus floor plans, digital twin (gitignored)
```

## Rules of thumb

- New logic goes in `src/minifsd` with a test in `tests/`. A ROS node only wraps
  it.
- `minifsd` never imports ROS.
- Anything large (datasets, renders, checkpoints, videos) goes in `data/` or
  `runs/`, which git ignores. Pre-commit blocks files over 1 MB.
- A choice someone might question gets an ADR in `docs/decisions/`.
