# 1. ROS 2, TF, Nav2 and the macOS setup

<!-- toc -->

## Contents

- [1. ROS 2 Jazzy core](#1-ros-2-jazzy-core)
  - [Big picture](#big-picture)
  - [In the weeds](#in-the-weeds)
  - [Gotchas](#gotchas)
  - [Try it (15 min)](#try-it-15-min)
  - [Read](#read)
- [2. TF2 and frames](#2-tf2-and-frames)
  - [Big picture](#big-picture-1)
  - [In the weeds](#in-the-weeds-1)
  - [Gotchas](#gotchas-1)
  - [Try it (15 min)](#try-it-15-min-1)
  - [Read](#read-1)
- [3. Message types this project uses](#3-message-types-this-project-uses)
  - [Big picture](#big-picture-2)
  - [In the weeds](#in-the-weeds-2)
  - [Gotchas](#gotchas-2)
  - [Try it (10 min)](#try-it-10-min)
  - [Read](#read-2)
- [4. Nav2 (Jazzy)](#4-nav2-jazzy)
  - [Big picture](#big-picture-3)
  - [In the weeds](#in-the-weeds-3)
  - [Gotchas](#gotchas-3)
  - [Try it (20 min)](#try-it-20-min)
  - [Read](#read-3)
- [5. RoboStack and pixi on macOS (osx-arm64)](#5-robostack-and-pixi-on-macos-osx-arm64)
  - [Big picture](#big-picture-4)
  - [Availability on robostack-jazzy for osx-arm64 (checked 2026-11-17)](#availability-on-robostack-jazzy-for-osx-arm64-checked-2026-11-17)
  - [In the weeds](#in-the-weeds-4)
  - [Gotchas](#gotchas-4)
  - [Try it (20 min)](#try-it-20-min-1)
  - [Read](#read-4)
- [6. Visualization: RViz2 and Foxglove](#6-visualization-rviz2-and-foxglove)
  - [Big picture](#big-picture-5)
  - [In the weeds](#in-the-weeds-5)
  - [Gotchas](#gotchas-5)
  - [Try it (15 min)](#try-it-15-min-2)
  - [Read](#read-5)

<!-- tocstop -->

Facts here were checked on 2026-11-17 against the Jazzy docs, the Nav2 Jazzy
docs, the RoboStack getting-started page, and the robostack-jazzy package
listings on anaconda.org. Two things changed recently and are easy to trip on.
Nav2's docs moved to versioned URLs (`docs.nav2.org/jazzy/...`), so old links
like `docs.nav2.org/configuration/...` now return 404. RoboStack's
getting-started page now uses Lyrical as its main example. Jazzy is still
supported, and the steps are the same with `jazzy` swapped in.

______________________________________________________________________

## 1. ROS 2 Jazzy core

**Why it matters for Mini FSD.** Every box in the architecture (sim_node,
perception_node, mission_node, Nav2) is a ROS node that talks over topics,
services, and actions. You need this from day one of Week 9, and you will lean
on the CLI tools every week after.

### Big picture

A ROS 2 system is a set of separate programs called **nodes**. Each node does
one job. Nodes find each other on the network automatically (through DDS, the
transport layer underneath) and talk in three ways:

- **Topics.** One-way streams. A publisher sends messages, any number of
  subscribers get them. Camera frames, odometry, and velocity commands are
  topics.
- **Services.** A request and a single reply. Good for quick queries like "clear
  the costmap."
- **Actions.** A long-running request with progress feedback and the option to
  cancel. "Drive to this pose" is an action, because it takes seconds and you
  want updates.

Each node also has **parameters**: named settings like `use_sim_time` or
`max_speed` that you set at launch and can read or change at runtime.

Your code runs inside **callbacks**. You never write a main loop that polls a
socket. You register "call this function when a message arrives" or "call this
every 50 ms," then hand the node to an **executor**, which waits for events and
runs the callbacks.

Code lives in **packages**, packages live in a **workspace**, and **colcon**
builds the workspace. A **launch file** starts many nodes at once with their
parameters.

### In the weeds

**A minimal rclpy node** (Python client library):

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class Perception(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("max_range", 3.0)
        self.sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.on_image,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

    def on_image(self, msg: Image):
        rng = self.get_parameter("max_range").value
        self.get_logger().info(
            f"{msg.width}x{msg.height} at {msg.header.stamp.sec}"
        )

    def tick(self):
        pass


def main():
    rclpy.init()
    rclpy.spin(Perception())  # single-threaded executor under the hood
    rclpy.shutdown()
```

**QoS (quality of service).** Every publisher and subscriber has a QoS profile.
The settings that matter:

- *Reliability*: `RELIABLE` (resend lost messages) or `BEST_EFFORT` (drop them).
- *History and depth*: `KEEP_LAST` with depth N keeps only the newest N
  messages.
- *Durability*: `VOLATILE` (late joiners get nothing old) or `TRANSIENT_LOCAL`
  (late joiners get the last message). Maps and `/tf_static` use transient
  local.

`qos_profile_sensor_data` is best effort, keep last, depth 5, volatile. Use it
for camera images. If inference is slow you want old frames dropped, not queued.
For perception you may even want depth 1, so you always process the newest
frame.

The compatibility rule that bites people: a `RELIABLE` subscriber will not
connect to a `BEST_EFFORT` publisher. The reverse works. If a topic shows up in
`ros2 topic list` but your callback never fires, check QoS first with
`ros2 topic info -v /topic`.

**Executors and callback groups.** `rclpy.spin(node)` uses a single-threaded
executor, so callbacks run one at a time. If your image callback takes 40 ms,
your timer waits. That's usually fine and much easier to reason about. If you
need parallel callbacks, use `MultiThreadedExecutor` plus a
`ReentrantCallbackGroup` or separate `MutuallyExclusiveCallbackGroup`s. Python's
GIL limits the gain, though ONNX Runtime releases the GIL during inference.

**Sim time.** The simulator owns time. sim_node publishes `rosgraph_msgs/Clock`
on `/clock` from MuJoCo's `data.time`. Every other node gets
`use_sim_time:=True`, which makes `self.get_clock().now()` return sim time. This
matters because the M1 may run the sim slower than real time when rendering.
With sim time, Nav2's timeouts and the controller rate stay consistent with the
world. Stamp every message (images, odom, TF) with the sim time of the step that
produced it.

**Lifecycle (managed) nodes.** A lifecycle node has states: unconfigured,
inactive, active, finalized. Transitions (`configure`, `activate`, `deactivate`,
`cleanup`) are explicit. Nav2's servers are all lifecycle nodes, and
`lifecycle_manager` walks them through configure then activate in order. Your
own nodes can be plain nodes. Commands:

```bash
ros2 lifecycle nodes
ros2 lifecycle get /planner_server
ros2 lifecycle set /planner_server activate
```

**Packages and workspaces.** Use `ament_python` for your Python nodes:

```bash
mkdir -p ~/minifsd_ws/src && cd ~/minifsd_ws/src
ros2 pkg create --build-type ament_python --node-name sim_node minifsd_sim
cd .. && colcon build --symlink-install
source install/setup.bash     # or setup.zsh in a plain zsh shell
```

Entry points go in `setup.py` under `console_scripts`. Launch and config files
must be listed in `data_files` or they won't be installed. `--symlink-install`
means Python edits take effect without rebuilding. New files and changed
`setup.py` still need a rebuild.

An **overlay** is your workspace sourced on top of an **underlay** (the ROS
install). With pixi, the underlay is the pixi environment itself, and you only
source your workspace's `install/setup.*`.

**Python launch files.**

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim = LaunchConfiguration("use_sim_time")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(package="minifsd_sim", executable="sim_node", output="screen"),
            Node(
                package="minifsd_perception",
                executable="perception_node",
                parameters=[{"use_sim_time": sim}, "config/perception.yaml"],
                remappings=[("image", "/camera/color/image_raw")],
            ),
        ]
    )
```

To include Nav2, use `IncludeLaunchDescription` on `nav2_bringup`'s
`navigation_launch.py` and pass `use_sim_time` and `params_file`.

**CLI introspection.** These are the commands you'll type most:

```bash
ros2 node list
ros2 node info /perception_node          # its pubs, subs, services, actions
ros2 topic list -t                       # with types
ros2 topic hz /camera/color/image_raw    # actual rate
ros2 topic echo /odom --once
ros2 topic echo /camera/color/image_raw --no-arr   # header only, skip pixel array
ros2 topic info -v /camera/color/image_raw         # QoS of every endpoint
ros2 param list /controller_server
ros2 param get /controller_server FollowPath.vx_max
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_tools view_frames           # writes a PDF of the TF tree
ros2 run rqt_graph rqt_graph
```

### Gotchas

- `ros2 topic hz` on large images can report a lower rate than the real one,
  because the CLI itself struggles to deserialize them in Python. Cross-check
  with a counter in your node.
- `ROS_LOCALHOST_ONLY` is deprecated in Jazzy. To keep traffic on your laptop,
  set `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`. Set a `ROS_DOMAIN_ID` too if
  anyone else on the campus Wi-Fi runs ROS.
- If `ros2 topic list` shows stale or missing topics, run `ros2 daemon stop` and
  try again. The CLI caches the graph in a background daemon.
- A launch file that runs fine but whose nodes all use wall time is a common
  bug. `use_sim_time` has to reach every node, including RViz and Nav2.

### Try it (15 min)

Write a talker that publishes a `std_msgs/String` at 10 Hz with best-effort QoS.
Write a listener with default (reliable) QoS. Run both. Confirm the listener
gets nothing, then use `ros2 topic info -v /chatter` to see why. Switch the
listener to `qos_profile_sensor_data` and watch it connect.

### Read

- [Jazzy: About quality of service settings](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
- [Jazzy: Executors](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Executors.html)
- [Jazzy: Creating launch files](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)
- [ROS 2 design: Clock and time](https://design.ros2.org/articles/clock_and_time.html)

______________________________________________________________________

## 2. TF2 and frames

**Why it matters for Mini FSD.** Nav2 refuses to move without a correct TF tree.
The camera-to-costmap step in Week 12 is a chain of frame transforms, and the
MuJoCo camera axes differ from ROS's. You set up the tree in Week 9 and debug it
again in Week 12.

### Big picture

Every sensor reading lives in some coordinate frame. TF2 keeps a tree of frames
and the transform between each parent and child, stamped with a time. Any node
can ask "where was point P in frame A, in frame B, at time t?" and TF2 chains
the transforms for you.

Two REPs (ROS conventions) fix the meaning:

- **REP 103.** SI units (meters, radians, seconds). Right-handed axes. For a
  body frame: x forward, y left, z up. Camera *optical* frames are different: z
  forward (into the scene), x right, y down. Optical frames get the suffix
  `_optical`.
- **REP 105.** The standard mobile robot chain is `map -> odom -> base_link`.
  `odom` is smooth but drifts. `map` is globally correct but can jump. A
  localizer publishes `map -> odom`, and odometry publishes `odom -> base_link`.

### In the weeds

Your tree for Mini FSD:

```text
map --(static identity, since localization is out of scope)--> odom
odom --(dynamic, from MuJoCo true pose, ~50 Hz)--> base_link
base_link --(static)--> camera_link          (x forward, z up)
camera_link --(static)--> camera_optical_frame   (z forward, x right, y down)
```

Images and the PointCloud2 should carry
`header.frame_id = camera_optical_frame`.

**Static vs dynamic.** Static transforms never change and go on `/tf_static`,
published once with transient-local QoS. Dynamic transforms go on `/tf` every
time they change.

The simplest working setup skips `camera_link` and publishes one static
transform from `base_link` straight to the optical frame. From the command line:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.10 --y 0 --z 0.30 --roll -1.5708 --pitch 0 --yaw -1.5708 \
  --frame-id base_link --child-frame-id camera_optical_frame
```

That roll/yaw pair of -90° turns a body frame (x forward, z up) into an optical
frame (z forward, y down). In a launch file, run the same executable as a `Node`
with those `arguments`.

The dynamic `odom -> base_link` from sim_node:

```python
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

t = TransformStamped()
t.header.stamp = sim_stamp  # from MuJoCo time, not wall time
t.header.frame_id = "odom"
t.child_frame_id = "base_link"
t.transform.translation.x, t.transform.translation.y = x, y
t.transform.rotation.z, t.transform.rotation.w = (
    math.sin(yaw / 2),
    math.cos(yaw / 2),
)
self.tf_broadcaster.sendTransform(t)
```

**MuJoCo cameras.** MuJoCo follows the OpenGL convention: the camera looks down
its own −Z axis, +Y is up in the image, +X is right. The ROS optical frame looks
down +Z with +Y down. The two differ by a 180° rotation about X. If you place
the MuJoCo camera on the robot with its own orientation, publish the ROS optical
frame from the robot body using the REP 103 rule above, and use MuJoCo's camera
pose only to check that the two agree. Do not copy MuJoCo's camera quaternion
into TF.

### Gotchas

- Quaternions in ROS messages are (x, y, z, w). MuJoCo stores them as (w, x, y,
  z). Mixing these up gives a robot that spins in RViz but not in the sim.
- A transform stamped with wall time while Nav2 runs on sim time shows up as
  "extrapolation into the future" or "lookup would require extrapolation into
  the past" errors. Stamp TF with sim time.
- Publish `odom -> base_link` from one place only. Two publishers of the same
  transform make the robot flicker.
- Nav2's default params mention both `base_link` and `base_footprint` in
  different nodes. Pick one name and search the YAML for the other.

### Try it (15 min)

Start sim_node (or just the static publisher above plus a fake
`odom -> base_link`). Run `ros2 run tf2_tools view_frames` and open the PDF.
Then run `ros2 run tf2_ros tf2_echo base_link camera_optical_frame` and check
that the rotation matches the -90°/-90° you set. Put a point 1 m in front of the
camera in RViz and make sure it shows up in front of the robot, not above it.

### Read

- [REP 103: Standard units and coordinate conventions](https://www.ros.org/reps/rep-0103.html)
- [REP 105: Coordinate frames for mobile platforms](https://www.ros.org/reps/rep-0105.html)
- [Jazzy: tf2 tutorials](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Tf2-Main.html)
- [Nav2 Jazzy: Setting up transformations](https://docs.nav2.org/jazzy/configuration_and_development/first_time_robot_setup_guide/transformation/setup_transforms/)

______________________________________________________________________

## 3. Message types this project uses

**Why it matters for Mini FSD.** sim_node publishes images, camera info, and
odometry in Week 9. Week 10's data engine may record them. Week 12 depends on
getting PointCloud2 right. The Twist vs TwistStamped choice decides whether
teleop and Nav2 can drive your sim at all.

### Big picture

Messages are typed structs. Most sensor messages start with a `std_msgs/Header`
holding a timestamp and a `frame_id`. That header ties the data to a time and a
TF frame, which is how Nav2 knows where your points are.

### In the weeds

- **`sensor_msgs/Image`**: `height`, `width`, `encoding` (`rgb8`, `bgr8`,
  `32FC1` for float depth in meters, `16UC1` for millimeters), `step` (bytes per
  row), `data`. Convert with `cv_bridge`
  (`CvBridge().cv2_to_imgmsg(arr, 'rgb8')`) or fill the fields with numpy
  yourself.
- **`sensor_msgs/CameraInfo`**: intrinsics. `k` is the 3×3 matrix
  `[fx 0 cx; 0 fy cy; 0 0 1]`, `d` is distortion (empty or zeros in sim), `p` is
  the 3×4 projection. Publish it on `/camera/color/camera_info` with the same
  stamp as each image. For MuJoCo, `fy = (H/2) / tan(fovy/2)`, with `fovy` in
  radians from the camera's `fovy` attribute. With square pixels, `fx = fy`.
- **`sensor_msgs/PointCloud2`**: a packed binary blob with a `fields` list
  describing the layout. Build one in Python with
  `sensor_msgs_py.point_cloud2.create_cloud_xyz32(header, points_Nx3)`.
- **`nav_msgs/Odometry`**: `header.frame_id = odom`,
  `child_frame_id = base_link`, `pose` (in odom) and `twist` (in base_link),
  each with a 6×6 covariance. Nav2 reads the twist for velocity. The TF
  transform is separate. Odometry does not publish TF for you.
- **`geometry_msgs/Twist`**: `linear.x` (m/s forward) and `angular.z` (rad/s,
  counterclockwise). A diff-drive robot uses only these two.
- **`geometry_msgs/TwistStamped`**: the same plus a header.

**Twist vs TwistStamped in Jazzy.** Nav2 added a parameter,
`enable_stamped_cmd_vel`. In Jazzy it defaults to **false**, so Nav2 publishes
plain `Twist` on `/cmd_vel`. I checked this in the Jazzy source
(`nav2_util/twist_publisher.hpp` declares it `false`), and the Jazzy
`nav2_params.yaml` does not override it. The Nav2 maintainers held back the
switch because Jazzy is an LTS. The default flipped to `TwistStamped` in Kilted.
`teleop_twist_keyboard` also publishes `Twist` unless you pass
`-p stamped:=true`.

My suggestion: subscribe to `Twist` in sim_node and leave the defaults. If you
switch to stamped later, set `enable_stamped_cmd_vel: true` on every Nav2 node
that publishes or consumes cmd_vel (controller_server, behavior_server,
velocity_smoother, collision_monitor, docking_server), plus teleop.

Nav2's velocity pipeline in Jazzy is:

```text
controller_server -> cmd_vel_nav -> velocity_smoother -> cmd_vel_smoothed
  -> collision_monitor -> cmd_vel
```

That matters for the next section.

### Gotchas

- `ros2 topic echo` on an image floods the terminal. Use `--no-arr` or
  `--field header`.
- Check `step` when your image looks sheared. It must equal
  `width * bytes_per_pixel`.
- RoboStack's Jazzy builds use Python 3.12 and numpy 2 (you can see `np2py312`
  in the package file names). Any pip package you add, including `mujoco` and
  `onnxruntime`, has to match that interpreter.

### Try it (10 min)

In sim_node, publish one `CameraInfo` per image with the same stamp. Run
`ros2 topic echo /camera/color/camera_info --once` and check that `k[0]` (fx)
matches your hand calculation from `fovy` and image height.

### Read

- [Jazzy: sensor_msgs package (Image, CameraInfo, PointCloud2)](https://docs.ros.org/en/jazzy/p/sensor_msgs/)
- [Jazzy: nav_msgs/Odometry](https://docs.ros.org/en/jazzy/p/nav_msgs/msg/Odometry.html)
- [Nav2 TwistStamped migration notice (Open Robotics Discourse)](https://discourse.openrobotics.org/t/notice-nav2-migrated-to-twiststamped-to-replace-twist-for-cmd-vel-topics/40944)

______________________________________________________________________

## 4. Nav2 (Jazzy)

**Why it matters for Mini FSD.** Nav2 is what turns "go to room 214" into
motion. Week 11 is getting it to drive to a goal on the generated map. Week 12
is feeding it your perception cloud so it dodges what the camera sees. The
recovery behaviors are your backing-up story.

### Big picture

Nav2 is a set of servers, each a lifecycle node, coordinated by a behavior tree:

- **bt_navigator** accepts the `NavigateToPose` action and runs a behavior tree
  (BT) that decides what to do next.
- **planner_server** computes a global path from the robot to the goal on the
  global costmap. That's the Smac planner here.
- **controller_server** follows the path at ~20 Hz and avoids nearby obstacles
  using the local costmap. That's MPPI here. It outputs velocity commands.
- **behavior_server** runs recovery behaviors: spin, back up, wait, drive on
  heading.
- **costmap_2d** keeps two grids. The *global* costmap (frame `map`) covers the
  whole building. The *local* costmap (frame `odom`, rolling window) covers a
  few meters around the robot. Each is built from layers:
  - *static layer*: the walls from `map_server`
  - *obstacle layer* or *voxel layer*: marks obstacles from sensor data (your
    PointCloud2) and clears free space by ray tracing
  - *inflation layer*: spreads cost out from obstacles so paths keep their
    distance
- **map_server** loads your generated map (a PGM/PNG image plus a YAML with
  resolution and origin) and publishes it on `/map`.
- **lifecycle_manager** starts all of these in order.
- **velocity_smoother** and **collision_monitor** sit between the controller and
  the robot.

The BT is the brain. The default tree replans the path once per second, follows
it, and on failure runs recovery actions before trying again.

### In the weeds

**Sending a goal.** From the CLI:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 4.0, y: 2.0}, orientation: {w: 1.0}}}}" \
  --feedback
```

From mission_node, `nav2_simple_commander` saves writing an action client:

```python
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

nav = BasicNavigator()
nav.waitUntilNav2Active(localizer="map_server")  # no AMCL in this project
nav.goToPose(goal_pose_stamped)
while not nav.isTaskComplete():
    fb = nav.getFeedback()  # distance_remaining, recoveries, ...
result = nav.getResult()  # TaskResult.SUCCEEDED / FAILED / CANCELED
```

In Jazzy the signature is
`waitUntilNav2Active(navigator='bt_navigator', localizer='amcl')`, and with the
default it waits for AMCL and an initial pose forever. Passing
`localizer='map_server'` makes it wait for your map server instead (a lifecycle
node you do run) and skips the initial-pose wait. Don't pass `None`: the code
would try to wait on a node named `None`.

**Bringup without AMCL.** `nav2_bringup`'s `bringup_launch.py` starts
localization (AMCL) too. You don't want that. Run `navigation_launch.py` for the
navigation servers, start `map_server` with its own lifecycle manager, and
publish `map -> odom` as a static identity.

**Smac planner.** Two variants matter:

- `nav2_smac_planner::SmacPlanner2D`: A\* on the grid, 8-connected. For a round
  diff-drive robot this is the right default.
- `nav2_smac_planner::SmacPlannerHybrid`: Hybrid-A\* with
  `motion_model_for_search: DUBIN` or `REEDS_SHEPP` and a
  `minimum_turning_radius`. Use it for car-like robots.

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner::SmacPlanner2D"
      tolerance: 0.25
      allow_unknown: true
      max_planning_time: 1.0
```

Jazzy's default params file still uses NavFn, so you have to change this
yourself.

**MPPI controller.** MPPI (Model Predictive Path Integral) samples a batch of
noisy velocity sequences around the last best one, rolls each through a motion
model for `time_steps × model_dt` seconds, scores every rollout with *critics*,
and takes a softmax-weighted average as the new command. Critics are cost terms:
stay off high-cost cells (`CostCritic`), follow and align with the path
(`PathFollowCritic`, `PathAlignCritic`, `PathAngleCritic`), approach the goal
(`GoalCritic`, `GoalAngleCritic`), respect limits (`ConstraintCritic`), prefer
driving forward (`PreferForwardCritic`). The costs don't need to be convex or
differentiable, which is why sampling is used. That is your "nonlinear
optimization" talking point.

Key parameters (Jazzy defaults from the docs and the shipped YAML):

| Param                        | Default                         | Notes                                   |
| ---------------------------- | ------------------------------- | --------------------------------------- |
| `motion_model`               | `DiffDrive`                     | also `Omni`, `Ackermann`                |
| `time_steps`                 | 56                              | horizon = 56 × 0.05 = 2.8 s             |
| `model_dt`                   | 0.05                            | match the control period (20 Hz → 0.05) |
| `batch_size`                 | 1000 (doc), 2000 (shipped YAML) | more samples, more CPU                  |
| `iteration_count`            | 1                               | raise batch size instead                |
| `temperature` / `gamma`      | 0.3 / 0.015                     | softmax sharpness / control cost        |
| `vx_max`, `vx_min`, `wz_max` | 0.5, -0.35, 1.9                 | raise `vx_max` for the speed sweep      |
| `vx_std`, `wz_std`           | 0.2, 0.2 (YAML: 0.4)            | sampling noise                          |
| `controller_frequency`       | 20.0                            | on controller_server                    |

**Behavior trees and recovery.** The default Jazzy tree
(`navigate_to_pose_w_replanning_and_recovery.xml`) does this:

1. Replan at 1 Hz (`RateController` around `ComputePathToPose`). If planning
   fails, clear the global costmap and retry once.
1. `FollowPath` with the controller. If it fails, clear the local costmap and
   retry once.
1. If both still fail, go round-robin through recoveries: clear both costmaps,
   `Spin` 1.57 rad, `Wait` 5 s, `BackUp` 0.30 m at 0.15 m/s. The whole thing
   retries up to 6 times.

Count recoveries from the action feedback (`number_of_recoveries`) for your
"recoveries per run" metric.

**Feeding a PointCloud2 into the local costmap.** Replace the default `scan`
source:

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      global_frame: odom
      robot_base_frame: base_link
      rolling_window: true
      width: 6            # meters; see gotcha about speed
      height: 6
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: camera
        camera:
          topic: /perception/obstacles
          data_type: "PointCloud2"
          min_obstacle_height: 0.05    # drop floor points
          max_obstacle_height: 1.5     # source default is 0.0, so set it
          obstacle_max_range: 3.5
          raytrace_max_range: 4.0
          marking: true
          clearing: true
          expected_update_rate: 0.2    # warn if no cloud for 0.2 s
          observation_persistence: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.55
        cost_scaling_factor: 3.0
```

`VoxelLayer` (the Jazzy default for the local costmap) does the same in 3D and
can clear under overhangs. For a single forward camera, `ObstacleLayer` is
simpler and enough.

Clearing works by ray tracing from the sensor origin to each point, freeing the
cells along the way. If perception only publishes obstacle points, an obstacle
that moves away leaves its mark behind, because no ray passes through that spot.
One fix (a design suggestion, not something Nav2 does for you): add a second
source with `marking: false, clearing: true` that carries points at the far edge
of the visible floor in each image column, raised to a height inside the min/max
window.

### Gotchas

- **collision_monitor will hold the robot still.** Jazzy's
  `navigation_launch.py` starts it, and its default source is `scan`. If a
  source gets no data within `source_timeout` (default 2.0 s), the robot is
  stopped. Point it at your cloud (`type: pointcloud`, with
  `min_height`/`max_height`), or set `source_timeout: 0.0` while you bring
  things up.
- **The default local costmap is 3 × 3 m.** At 1.5 m/s, a 2.8 s MPPI horizon
  reaches 4.2 m, past the costmap edge. Widen the costmap, shorten the horizon,
  or both.
- **CPU on 8 GB M1.** MPPI at batch 2000 and 20 Hz, plus perception, plus MuJoCo
  rendering, will compete for cores. The docs suggest 1000 at 50 Hz or 2000 at
  30 Hz as sensible pairs. Turn off `visualize`, which the docs say slows MPPI a
  lot. Measure before you tune.
- Your odom must publish at least as fast as `controller_frequency`.
- A failed `configure` on one server (a bad map path, a typo in a plugin name)
  stalls the whole bringup. The lifecycle manager log names the node. Read it
  before anything else.

### Try it (20 min)

Draw a 10 × 10 m map in any paint program (black walls, white floor), save it as
PGM with a YAML (`resolution: 0.05`, `origin: [0, 0, 0]`). Start `map_server`
plus its lifecycle manager and look at `/map` in RViz. Then bring up
`navigation_launch.py` with a fake odom node that integrates `/cmd_vel` and
publishes `odom -> base_link`. Send one `NavigateToPose` goal from the CLI and
watch the path and velocity commands.

### Read

- [Nav2 Jazzy: Navigation concepts](https://docs.nav2.org/jazzy/getting_started/navigation_concepts/)
- [Nav2 Jazzy: MPPI controller](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/)
- [Nav2 Jazzy: Obstacle layer](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/obstacle/)
- [Nav2 Jazzy: Collision monitor](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/collision_monitor/configuring_collision_monitor_node/)

______________________________________________________________________

## 5. RoboStack and pixi on macOS (osx-arm64)

**Why it matters for Mini FSD.** This is the Week 9 spike. If the environment
isn't reproducible, nothing after it is. The design doc names RoboStack as the
main path and Docker as the fallback.

### Big picture

RoboStack repackages ROS as conda packages, built natively for Apple Silicon.
**pixi** is a fast package manager for conda packages. It keeps everything for
one project in a `pixi.toml` and an exact `pixi.lock`, installed into `.pixi/`
inside the project folder. `pixi shell` drops you into the environment.
`pixi run <cmd>` runs one command inside it. Nothing touches your system Python
or Homebrew.

### Availability on robostack-jazzy for osx-arm64 (checked 2026-11-17)

I queried the anaconda.org API for the `robostack-jazzy` channel. All of these
have osx-arm64 builds. The newest build for most was uploaded 2026-10-22 (build
string `h60d57d3_27`):

| Package                           | Version (osx-arm64) |
| --------------------------------- | ------------------- |
| `ros-jazzy-desktop`               | 0.11.0              |
| `ros-jazzy-navigation2`           | 1.3.13              |
| `ros-jazzy-nav2-bringup`          | 1.3.13              |
| `ros-jazzy-nav2-mppi-controller`  | 1.3.13              |
| `ros-jazzy-nav2-smac-planner`     | 1.3.13              |
| `ros-jazzy-foxglove-bridge`       | 3.6.0               |
| `ros-jazzy-rviz2`                 | 14.1.24             |
| `ros-jazzy-teleop-twist-keyboard` | 2.4.1               |

Also present for osx-arm64: `nav2-simple-commander`, `nav2-collision-monitor`,
`cv-bridge`, `sensor-msgs-py`, `tf2-tools`, `rqt-graph`, `rqt-tf-tree`,
`rqt-image-view`, `rosbag2`, `rmw-cyclonedds-cpp`, `rmw-zenoh-cpp`.

### In the weeds

Setup (pixi 0.84.2 is current as of Nov 10, 2026):

```bash
curl -fsSL https://pixi.sh/install.sh | bash      # then open a new terminal
pixi init minifsd_ws --channel https://prefix.dev/robostack-jazzy
cd minifsd_ws
pixi workspace platform add osx-latest=osx-arm64 --osx 14
pixi workspace platform rm osx-arm64
pixi add ros-jazzy-desktop ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
         ros-jazzy-foxglove-bridge ros-jazzy-teleop-twist-keyboard \
         ros-jazzy-rmw-cyclonedds-cpp \
         colcon-common-extensions compilers cmake pkg-config make ninja
```

The two `platform` lines matter. RoboStack's current builds need macOS 14+, but
pixi assumes macOS 13 unless told otherwise, and then it quietly installs older
builds.

Then in `pixi.toml`:

```toml
[activation]
scripts = ["install/setup.bash"]

[activation.env]
RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
ROS_AUTOMATIC_DISCOVERY_RANGE = "LOCALHOST"

[tasks]
build = "colcon build --symlink-install --cmake-args -DPython_FIND_VIRTUALENV=ONLY -DPython3_FIND_VIRTUALENV=ONLY"
sim = "ros2 launch minifsd_bringup sim.launch.py"
```

Workflow:

```bash
pixi run build          # build once so install/setup.bash exists
pixi shell              # now ros2, colcon, rviz2 are on PATH and the overlay is sourced
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Commit `pixi.toml` and `pixi.lock`. That lock file is your "pinned environment"
deliverable.

### Gotchas

- **Fast DDS on macOS.** RoboStack issue #50 reported that subscribers got no
  messages with the default Fast DDS on macOS. Cyclone DDS worked. A later
  comment says a rebuild fixed it, but issue #124 (closed) still reports Fast
  DDS thread-affinity errors on macOS 26 (Tahoe). Cyclone and Zenoh don't show
  them. Set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` as above and skip the
  problem.
- **RViz2.** Older Jazzy builds crashed RViz2 with a bus error on Apple Silicon
  (issue #26, closed July 2025). A fix for Ogre library paths on macOS was
  merged on 2026-11-05 (RoboStack PR #185). If you see
  `dyld: Library not loaded: @rpath/libOgreMain_rviz...`, run `pixi update` to
  pull the rebuilt packages. RViz2 may still print an error on exit. That's
  harmless.
- **SIP and DYLD.** macOS strips `DYLD_*` variables when a protected binary like
  `/bin/sh` starts a child process. That's why old source-build guides told
  people to disable SIP. Conda packages use embedded rpaths, so you should not
  need to touch SIP. Don't disable it.
- **rosdep doesn't work with pixi.** It tries to `conda install`. Add
  dependencies with `pixi add` and keep `package.xml` accurate for
  documentation.
- **Shells.** pixi runs activation scripts with bash, so `install/setup.bash` is
  right inside pixi. If you ever source by hand in zsh, use `install/setup.zsh`.
- **Don't mix environments.** Never source a different ROS install in the same
  shell. The `PYTHONPATH` it sets conflicts with pixi's.
- **Compilers.** You only need `compilers`, `cmake`, and `ninja` if you write
  C++ (the stretch Nav2 plugin). The conda `compilers` package brings clang for
  osx-arm64. You still need Apple's Command Line Tools for the SDK headers
  (`xcode-select --install`).
- **macOS firewall.** The first DDS run may bring up a macOS prompt asking
  whether Python can accept incoming connections. Allow it. I could not confirm
  that denying it breaks discovery on localhost, but allowing it removes one
  variable.
- **8 GB RAM.** RViz2, Nav2, MuJoCo with rendering, and perception together are
  tight. Watch Activity Monitor during the spike and write the numbers into your
  decision note.

### Try it (20 min)

Do the setup above in a fresh folder. Then run `ros2 run demo_nodes_cpp talker`
in one `pixi shell` and `ros2 run demo_nodes_py listener` in another. Then run
`rviz2`. Time the whole thing from `pixi init` to working RViz and put it in the
Week 9 decision note.

### Read

- [RoboStack: Getting started](https://robostack.github.io/GettingStarted.html)
- [pixi: ROS 2 tutorial](https://pixi.sh/latest/tutorials/ros2/) (uses Humble,
  but the workflow is the same)
- [robostack-jazzy: ros-jazzy-navigation2 package listing](https://anaconda.org/robostack-jazzy/ros-jazzy-navigation2)
- [RoboStack ros-jazzy issue #50 (Fast DDS on macOS)](https://github.com/RoboStack/ros-jazzy/issues/50)

______________________________________________________________________

## 6. Visualization: RViz2 and Foxglove

**Why it matters for Mini FSD.** The Week 9 deliverable is a screen recording of
the robot driving in RViz or Foxglove. In Weeks 11 and 12 you debug Nav2 and the
costmap by looking at them. The Week 13 demo videos come from here too.

### Big picture

Both tools subscribe to your topics and draw them in 3D. **RViz2** is the
classic ROS tool. It runs as a ROS node inside your pixi environment.
**Foxglove** is a separate desktop app (or Chrome web app) that connects to ROS
through **foxglove_bridge**, a node that serves your topics over a WebSocket.
Foxglove has nicer panels (plots, image overlays, raw message views) and doesn't
depend on Qt and OpenGL inside your ROS install.

### In the weeds

**RViz2.**

- *Fixed Frame* (top of Displays): the frame everything is drawn in. Use `odom`
  while testing the sim and `map` once Nav2 is up. If it's wrong or missing in
  TF, every display goes red.
- Add displays by topic (Add → By topic): `TF`, `Image`
  (`/camera/color/image_raw`), `Odometry`, `Map` (`/map`, plus
  `/local_costmap/costmap` and `/global_costmap/costmap` with Color Scheme set
  to "costmap"), `PointCloud2` (`/perception/obstacles`), `Path` (`/plan`).
- Set QoS per display. Image and PointCloud2 displays have a Reliability setting
  under Topic. Set it to Best Effort for sensor topics.
- Use the "Nav2 Goal" tool button to send NavigateToPose goals by clicking.
  `nav2_bringup` ships an RViz config with this set up.
- Save your setup as `minifsd.rviz` in your bringup package and launch with
  `rviz2 -d <path>`, with `use_sim_time:=true`.

**Foxglove.**

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml   # default port 8765
```

In the app: Open connection → Foxglove WebSocket → `ws://localhost:8765`. Add a
3D panel (set the display frame like RViz's fixed frame), an Image panel, and a
Plot panel for `/cmd_vel/linear/x` against `/odom/twist/twist/linear/x`. The
bridge subscribes to every topic by default. On an 8 GB machine, set
`topic_whitelist` (a list of regexes, default `[".*"]`) to skip what you don't
need, such as ground-truth depth. For best-effort sensor topics, the bridge
picks best effort automatically if any publisher uses it.

### Gotchas

- Foxglove needs a free account to sign in to the app. The pricing page lists a
  free plan. Offline use without signing in is an Enterprise feature.
- The Foxglove ROS docs install the bridge with `apt`. On the Mac, use
  `pixi add ros-jazzy-foxglove-bridge` instead. Version 3.6.0 is on
  robostack-jazzy for osx-arm64.
- The bridge code now lives in the `foxglove/foxglove-sdk` repo, not the old
  `ros-foxglove-bridge` repo. Search there for issues.
- If RViz2 is unstable on your machine (see section 5), Foxglove is the
  fallback. It runs outside the ROS environment, so a Qt or Ogre problem can't
  break it.
- Streaming raw 320×240 RGB at 20 Hz is fine. Streaming uncompressed depth plus
  several costmaps at full rate through the bridge eats CPU you need for
  perception. Throttle or skip during benchmark runs, and never visualize during
  latency measurements.

### Try it (15 min)

With the sim running, open RViz2 with fixed frame `odom` and add TF, Image, and
Odometry. Drive with teleop and record 20 seconds (macOS Cmd+Shift+5). Then
close RViz, start foxglove_bridge, and build the same view in Foxglove with a
speed plot. Note which one used less CPU in Activity Monitor and write that into
the Week 9 decision note.

### Read

- [Jazzy: RViz user guide](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)
- [Foxglove: ROS 2 getting started](https://docs.foxglove.dev/docs/getting-started/frameworks/ros2)
- [foxglove_bridge README (configuration)](https://github.com/foxglove/foxglove-sdk/blob/main/ros/src/foxglove_bridge/README.md)
