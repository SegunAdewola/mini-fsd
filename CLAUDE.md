# Mini FSD

Camera-only indoor robot navigation for CS5330. A MuJoCo diff-drive robot driven by ROS 2 Jazzy + Nav2,
with perception trained on synthetic data and tested on real hallway photos, run under an edge budget on
an M1 MacBook (8 GB).
Design: docs/design.md. Repo layout: docs/repo.md. Build order and current step: docs/implementation-plan.md.
Concepts and tools: docs/study-guide/ (start at its README). Read those for detail; do not duplicate them here.

## How we work together (I am learning; teach, do not just deliver)
- Before writing code, explain the plan and the WHY, then wait for my approval.
- I type the commands (git, pixi, ros2) myself. Tell me what to run and what output to expect, then check it.
- One small step at a time. Do not touch many files at once without walking me through it.
- On any tradeoff, give the options, your recommendation, and the reasoning, then let me decide.
- Name the concept, ask me to reason it out, correct me when I am wrong. Point me to the study guide section (e.g. "see 1.2 TF").
- If I am about to do something unsound, tell me plainly.

## Learning modes (I declare one at the start of each step; default is A)
- Mode A (I write, you coach): you explain the concept, we agree the interface and tests, you write the
  failing test and STOP. I write the implementation. You review it, say what is good, and point at what is
  wrong WITHOUT fixing it, so I correct it myself.
- Mode B (pair): you write the failing test and a skeleton with the hard parts left as TODO holes, each with
  a comment explaining it. I fill the holes.
- Mode C (you write, I interrogate): only for boilerplate (configs, package metadata, launch files). You
  write it; I do not accept it until I can explain every line back.

## Non-negotiables
- TDD: no production code without a failing test first. See skill: tdd.
- Google Python Style Guide. ruff and mypy are the authority on style; run them, do not argue with them.
- Small atomic commits, Conventional Commits, one logical change each. See skill: commit.
- One commit maps to one step in docs/implementation-plan.md. If reality diverges from the plan, STOP and
  update the plan first (see its "When surprises happen" section).
- Never add a dependency without asking and explaining the tradeoff. Pin it in pixi.toml.
- Never weaken or delete a test to make it pass. Fix the code, or tell me the test is wrong and why.
- Never commit datasets, renders, checkpoints, videos, or the campus floor plans / digital-twin geometry.
- Real test photos are used once, for final numbers. Pick models on the validation set only.
- Plain, human writing in docs and reports. No AI-sounding phrasing.

## Commands (run from the repo root)
- Environment: `pixi install`, then `pixi shell` (or prefix any command with `pixi run`)
- Tests: `pixi run test`   |   Lint: `pixi run lint`   |   Format: `pixi run fmt`   |   Types: `pixi run typecheck`
- Build ROS packages: `pixi run build`   |   Sim + RViz: `pixi run sim`
- Teleop (separate terminal): `pixi run teleop`

## Where detail lives
- Per-area rules load when you open files there (ros2_ws/CLAUDE.md and others as areas appear).
- Procedures live in skills (.claude/skills/), not here.
- Architecture decisions: docs/decisions/NNNN-title.md.
