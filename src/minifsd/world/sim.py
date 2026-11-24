"""MuJoCo simulator for the planar robot (ADR 0002).

The ROS node (R5) wraps this class: /cmd_vel calls command(), a timer calls
step(), and the results are published as /odom, /tf and camera images. All
the logic lives here, so it is tested without ROS (ADR 0003).

Time is simulation time, not wall-clock time, so a test that steps 1 s
always sees the same result however fast the machine is.
"""

import math
from collections.abc import Sequence

import mujoco
import numpy as np

from minifsd.world import clutter as clutter_lib
from minifsd.world import floor_plan, mjcf

# Stop if no command arrives for this long. teleop and Nav2 both resend
# commands at 10 Hz or more, so 0.3 s means 3 missed messages. With the
# actuators' 0.05 s time constant, the robot is stopped within 0.5 s of the
# last command.
WATCHDOG_TIMEOUT = 0.3


def body_to_world(v: float, w: float, yaw: float) -> tuple[float, float, float]:
    """Converts a unicycle command to world-frame joint velocities.

    Args:
        v: Forward speed along the robot's heading, m/s.
        w: Turn rate, rad/s, counterclockwise.
        yaw: Current heading, radians.

    Returns:
        (x_dot, y_dot, yaw_dot) for the base_x, base_y and base_yaw joints.
    """
    return v * math.cos(yaw), v * math.sin(yaw), w


class Sim:
    """The robot in one floor plan.

    Example, the way the ROS node uses it:
        s = Sim(plan)
        s.command(0.5, 0.0)   # on every /cmd_vel message
        s.step(0.05)          # on a 20 Hz timer
        s.pose(), s.render_rgb()
    """

    def __init__(
        self,
        plan: floor_plan.FloorPlan,
        watchdog_timeout: float = WATCHDOG_TIMEOUT,
        clutter: Sequence[clutter_lib.Box] = (),
    ) -> None:
        """Builds the model and puts the robot at the plan's spawn pose.

        Args:
            plan: The floor plan to simulate.
            watchdog_timeout: Seconds without a command before stopping.
            clutter: Boxes in the world but not in the floor plan.
        """
        self.plan = plan
        self.clutter = tuple(clutter)
        self.model = mujoco.MjModel.from_xml_string(
            mjcf.build_mjcf(plan, self.clutter)
        )
        self.data = mujoco.MjData(self.model)
        spawn = self.model.key(mjcf.SPAWN_KEY).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, spawn)
        mujoco.mj_forward(self.model, self.data)

        self._qpos = [self.model.joint(n).qposadr[0] for n in mjcf.JOINTS]
        self._qvel = [self.model.joint(n).dofadr[0] for n in mjcf.JOINTS]
        self._ctrl = [self.model.actuator(n).id for n in mjcf.ACTUATORS]

        self._watchdog_timeout = watchdog_timeout
        self._v = 0.0
        self._w = 0.0
        self._last_command_time = -math.inf  # no command yet: stay still
        self._renderer: mujoco.Renderer | None = None
        self._floor_geom = self.model.geom("floor").id

    @property
    def time(self) -> float:
        """Simulation time in seconds since the start."""
        return float(self.data.time)

    def command(self, v: float, w: float) -> None:
        """Sets the forward speed and turn rate, clamped to the limits.

        The command holds until the next one, or until the watchdog fires.

        Args:
            v: Forward speed, m/s.
            w: Turn rate, rad/s, counterclockwise.
        """
        self._v = float(np.clip(v, -mjcf.MAX_SPEED, mjcf.MAX_SPEED))
        self._w = float(np.clip(w, -mjcf.MAX_TURN, mjcf.MAX_TURN))
        self._last_command_time = self.time

    def step(self, seconds: float) -> None:
        """Advances the simulation.

        Args:
            seconds: How long to simulate. Rounded to whole physics steps
                (2 ms each).
        """
        n = max(1, round(seconds / self.model.opt.timestep))
        for _ in range(n):
            stale = self.time - self._last_command_time > self._watchdog_timeout
            v, w = (0.0, 0.0) if stale else (self._v, self._w)
            # Re-aim the x/y velocity every physics step: the heading changes
            # as the robot turns, and the command is along the heading.
            yaw = float(self.data.qpos[self._qpos[2]])
            for ctrl, value in zip(
                self._ctrl, body_to_world(v, w, yaw), strict=True
            ):
                self.data.ctrl[ctrl] = value
            mujoco.mj_step(self.model, self.data)

    def set_pose(self, pose: floor_plan.Pose2D) -> None:
        """Puts the robot at a pose, at rest, without stepping physics.

        For rendering training frames (V7); driving uses command() and step().

        Args:
            pose: The new pose in the world frame.
        """
        for i, value in zip(
            self._qpos, (pose.x, pose.y, pose.yaw), strict=True
        ):
            self.data.qpos[i] = value
        self.data.qvel[:] = 0.0
        self._v = self._w = 0.0
        mujoco.mj_forward(self.model, self.data)  # update positions, cameras

    def pose(self) -> floor_plan.Pose2D:
        """The robot's true pose in the world frame (ADR 0004)."""
        x, y, yaw = (float(self.data.qpos[i]) for i in self._qpos)
        return floor_plan.Pose2D(x, y, yaw)

    def velocity(self) -> tuple[float, float]:
        """The robot's actual (forward speed, turn rate), in its own frame."""
        vx, vy, w = (float(self.data.qvel[i]) for i in self._qvel)
        yaw = self.pose().yaw
        # Project the world velocity onto the heading.
        return vx * math.cos(yaw) + vy * math.sin(yaw), w

    def render_rgb(self) -> np.ndarray:
        """The front camera image: (H, W, 3) uint8."""
        r = self._get_renderer()
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=mjcf.CAMERA)
        return np.asarray(r.render(), dtype=np.uint8)

    def render_depth(self) -> np.ndarray:
        """Depth from the front camera: (H, W) float32, metres.

        Depth is distance along the camera's viewing axis, the same
        convention as NYUv2 and the depth models.
        """
        r = self._get_renderer()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=mjcf.CAMERA)
        depth = np.asarray(r.render(), dtype=np.float32)
        r.disable_depth_rendering()
        return depth

    def render_floor_mask(self) -> np.ndarray:
        """Which camera pixels show the floor: (H, W) bool.

        Uses MuJoCo's segmentation rendering, which reports the geom behind
        every pixel, so the mask is exact.
        """
        r = self._get_renderer()
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=mjcf.CAMERA)
        seg = np.asarray(r.render())  # (H, W, 2): object id, object type
        r.disable_segmentation_rendering()
        is_floor = (seg[..., 0] == self._floor_geom) & (
            seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        )
        return np.asarray(is_floor, dtype=bool)

    def close(self) -> None:
        """Frees the renderer's OpenGL context."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_renderer(self) -> mujoco.Renderer:
        # Created on first use, so tests that only drive need no OpenGL.
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH
            )
        return self._renderer
