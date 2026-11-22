# ADR 0001: Install ROS 2, MuJoCo and tools with pixi and RoboStack

- Status: Accepted
- Date: 2026-11-22
- Plan step: 0.4

## Context

The project needs ROS 2 Jazzy, Nav2, RViz, MuJoCo and the Python test tools on
an M1 MacBook Air with 8 GB of RAM, running macOS 26.5. ROS 2 has no official
binary packages for macOS. Building it from source takes hours and breaks often.
The same environment also has to be rebuilt in CI and, without the ROS parts, on
the Explorer cluster and Colab.

RoboStack repackages ROS 2 as conda packages for macOS on Apple Silicon. pixi is
a project-level package manager for conda (and PyPI) packages: it installs into
`.pixi/` inside the repo, writes every exact version to `pixi.lock`, and runs
commands with `pixi run`.

Spike results on this machine (2026-11-22, pixi 0.84.2):

| Check                                      | Result                               |
| ------------------------------------------ | ------------------------------------ |
| `demo_nodes_cpp` talker to `demo_nodes_py` | Messages received, Cyclone DDS       |
| `rviz2`                                    | Opens and runs smoothly (OpenGL 2.1) |
| `ros2 pkg prefix nav2_bringup`             | Found inside `.pixi/envs/default`    |
| MuJoCo offscreen render, 320×240           | Works, about 2 ms per frame          |
| Environment size (`du -sh .pixi`)          | 6.4 GB                               |

Resolved versions: Python 3.12.16, ROS 2 Jazzy desktop 0.11.0, Navigation2
1.3.13, rmw_cyclonedds_cpp 2.2.4, MuJoCo 4.0.1, NumPy 2.5.5.

## Decision

Use pixi with the `robostack-jazzy` and `conda-forge` channels for the whole
environment. Commit `pixi.lock`. Run every tool through `pixi run` or inside
`pixi shell`. Use Cyclone DDS restricted to localhost, set in `pixi.toml`.

MuJoCo comes from PyPI through pixi, pinned to 4.0.1 or later, because 4.0.0 has
a solver bug.

## Consequences

- One command (`pixi install`) rebuilds the same environment on another Mac or
  in CI, and `pixi.lock` makes it exact.
- Nothing is installed system-wide, so the conda `base` environment and Homebrew
  Python are untouched. The flip side is that tools only exist inside the pixi
  environment: `ros2` fails with "command not found" in a plain terminal, and
  the editor must point at `.pixi/envs/default/bin/python`.
- The environment stores absolute paths. Moving the repo means a fresh clone and
  `pixi install`, not a folder move.
- 6.4 GB of disk on a machine with little free space.
- RoboStack's current builds need macOS 14 or later, recorded as the platform
  floor in `pixi.toml`.
- The Nav2 `map_server` check from the original spike plan moved to step R8,
  where the generated map is served for real.

## Alternatives considered

- **Build ROS 2 from source on macOS.** The official route, but slow and fragile
  on Apple Silicon, and it would need repeating on every machine.
- **Docker or a Linux VM.** The usual way to run ROS on a Mac. With 8 GB of RAM,
  a VM plus RViz plus MuJoCo leaves too little memory, GUI forwarding is clumsy,
  and latency numbers measured inside a VM would not match the native edge
  budget.
- **conda or mamba with an `environment.yml`.** Same RoboStack packages, but no
  built-in lockfile or task runner, and it encourages a shared global
  environment.
- **uv with a venv.** Only installs PyPI packages, and ROS 2 is not on PyPI. uv
  still fits the parts that are pure Python (Explorer, Colab).
