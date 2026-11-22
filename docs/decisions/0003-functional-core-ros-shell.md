# ADR 0003: Keep all ROS-free logic in one plain Python package

- Status: Accepted
- Date: 2026-11-21
- Plan step: 0.5

## Context

The same logic is needed in four places: the ROS nodes on the M1, the data
engine that renders training frames, the training and eval scripts on Explorer
and Colab, and the Dec 15 demo script, which must run without ROS. ROS nodes
are slow to test: they need a ROS environment, an executor, and topics.
Explorer and Colab have no ROS at all.

## Decision

All logic that does not need ROS lives in `src/minifsd/`, a normal Python
package installed in editable mode into the pixi environment. This includes the
building generator, MJCF builder, simulator wrapper, geometry, perception,
metrics, data export and mission parsing. It is tested with plain pytest and
Hypothesis in `tests/`.

The ROS packages in `ros2_ws/src/` are thin shells. They read parameters,
convert messages to and from numpy, call `minifsd`, and publish. They hold no
algorithms. This pattern is often called "functional core, imperative shell".

## Consequences

- Most tests run in well under a second with no ROS, so TDD stays fast.
- Explorer, Colab and the demo script install `minifsd` without ROS.
- `minifsd` must never import `rclpy` or any ROS message type. Review checks
  this; a lint rule can enforce it later.
- Each ROS node gets one in-process smoke test that checks the wiring, not the
  logic.

## Alternatives considered

- **Logic inside each ROS package.** The usual ROS layout. Tests need ROS, and
  the code can't be reused on Explorer or in the demo without copying.
- **Several small packages** (one per area). More packaging overhead for a
  solo, five-week project. Subpackages inside `minifsd` give the same
  separation.
