# ADR 0002: Model the robot as a planar base, not a wheeled one

- Status: Accepted
- Date: 2026-11-23
- Plan step: R2

## Context

The robot only needs to do what Nav2 asks: drive forward at speed `v` and turn
at rate `w` (a unicycle command on `/cmd_vel`). The experiments measure
perception and planning, not wheel dynamics. The hours budget is 10 to 15 a
week, with about half going to computer vision.

A wheeled model in MuJoCo needs wheel bodies, hinge joints, tyre friction, a
caster, and tuning so the wheels don't slip or bounce. Speed tracking then
depends on that tuning, and the "maximum safe speed against perception latency"
sweep would partly measure wheel slip.

## Decision

The base is one body with three joints: slide along world x, slide along world
y, and a hinge about z. Each joint has a velocity actuator. The simulator turns
`(v, w)` into joint velocities:

```text
x_dot = v cos(yaw),  y_dot = v sin(yaw),  yaw_dot = w
```

The base has no vertical joint and the floor has no contacts, so it cannot tip
or drag. It still collides with walls. Gains are set for a 0.05 s time constant
(mass 10 kg, kv 200).

## Consequences

- Speed tracking is close to exact, so the speed sweep measures perception and
  control latency, which is what it is meant to measure.
- The robot can't slip sideways, because the simulator only ever commands motion
  along its heading. A sideways command is still possible in principle, so the
  simulator is the one place that enforces the unicycle model.
- No wheel odometry error. Combined with ADR 0004 (true pose), odometry is
  perfect. The report states both in its limitations.
- Results won't carry over directly to a real robot with wheels. Swapping in a
  wheeled base later changes only `mjcf.py` and the command mapping in the
  simulator; the ROS interface (`/cmd_vel` in, `/odom` out) stays the same.

## Alternatives considered

- **Two driven wheels plus a caster.** The realistic option, and what a real
  diff-drive robot is. Costs tuning time and adds slip to every speed result.
- **`mujoco_ros2_control` with a diff-drive controller.** The standard ROS way,
  but it may not be packaged for macOS arm64 (design doc), and it brings the
  wheel tuning with it.
- **Teleport the base each step** (set the pose directly, no physics). Exact,
  but the robot would pass through walls, and collisions are needed for the
  collision rate metric.
