# ADR 0004: Use the simulator's true pose instead of localization

- Status: Accepted
- Date: 2026-11-21
- Plan step: R8

## Context

Nav2 needs the robot's pose in the `map` frame. A real robot estimates it with
localization (AMCL against a map, or SLAM), usually from LiDAR. This project is
camera-only and about 50% of the time must go to computer vision. Localization
from a single camera is a research project of its own.

## Decision

`sim_node` publishes `odom -> base_link` from the simulator's true pose, and a
static identity transform publishes `map -> odom`. The map is generated from
the same building layout the simulator uses, so the map and the world match
exactly.

## Consequences

- Navigation errors come from perception, planning and control only, which is
  what the experiments measure.
- Results overstate real-world performance. The report states this assumption
  in its limitations section.
- Swapping in real localization later only changes who publishes
  `map -> odom`; no other node changes.

## Alternatives considered

- **AMCL with a simulated laser.** Adds a LiDAR the design rules out.
- **Visual odometry or visual SLAM.** Real CV work, but large, and it would
  pull time away from the perception experiments the course grades.
