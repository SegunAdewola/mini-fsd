"""Starts the simulator, Nav2 on the floor plan's map, and RViz.

    ros2 launch mini_fsd_bringup nav.launch.py seed:=3 clutter:=6 people:=2

clutter:=N puts N boxes in the world that are not on the map; the robot must
see them with the camera to get around them. people:=N (0 to 2) adds walkers
pacing the hallway (M4.7). perception:=model runs the trained models on the
camera image instead of reading the sim's exact depth (M4.6).

Send a goal with RViz's "2D Goal Pose" tool, or by room number:
    ros2 topic pub --once /mission std_msgs/String "data: 'go to room 103'"

Localization is the sim's true pose (ADR 0004): map -> odom is a static
identity and there is no AMCL.
"""

import pathlib
import tempfile

import launch
import launch.actions
import launch.conditions
import launch.launch_description_sources
import launch.substitutions
import launch_ros.actions
import launch_ros.parameter_descriptions
from ament_index_python.packages import get_package_share_directory

import minifsd
from minifsd.world import floor_plan, occupancy

SHARE = pathlib.Path(get_package_share_directory("mini_fsd_bringup"))
PARAMS = str(SHARE / "config" / "nav2.yaml")
SIM_TIME = {"use_sim_time": True}
# The exported models (M4.4): src/minifsd/__init__.py -> repo root.
MODELS = pathlib.Path(minifsd.__file__).resolve().parents[2] / "data/models"
# Started and kept alive by the lifecycle manager, in this order.
NAV2_NODES = [
    ("nav2_map_server", "map_server"),
    ("nav2_planner", "planner_server"),
    ("nav2_controller", "controller_server"),
    ("nav2_behaviors", "behavior_server"),
    ("nav2_bt_navigator", "bt_navigator"),
]


def write_map(seed: int, rooms_per_side: int) -> pathlib.Path:
    """Writes the map for the floor plan the sim builds; returns its YAML."""
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=rooms_per_side)
    out = pathlib.Path(tempfile.mkdtemp(prefix="minifsd_map_"))
    return occupancy.save(occupancy.from_floor_plan(plan), out / f"seed_{seed}")


def _map_server(
    context: launch.LaunchContext,
) -> list[launch.Action]:
    # Runs at launch time, once the seed argument has a value, so the map is
    # always the floor plan the sim builds from the same seed.
    yaml_path = write_map(
        int(context.launch_configurations["seed"]),
        int(context.launch_configurations["rooms_per_side"]),
    )
    return [
        launch_ros.actions.Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[PARAMS, SIM_TIME, {"yaml_filename": str(yaml_path)}],
        )
    ]


def as_int(
    value: launch.substitutions.LaunchConfiguration,
) -> launch_ros.parameter_descriptions.ParameterValue:
    """Launch arguments are strings; the nodes declare these as integers."""
    return launch_ros.parameter_descriptions.ParameterValue(
        value, value_type=int
    )


def generate_launch_description() -> launch.LaunchDescription:
    seed = launch.substitutions.LaunchConfiguration("seed")
    rooms = launch.substitutions.LaunchConfiguration("rooms_per_side")
    clutter = launch.substitutions.LaunchConfiguration("clutter")
    people = launch.substitutions.LaunchConfiguration("people")
    rviz = launch.substitutions.LaunchConfiguration("rviz")
    perception = launch.substitutions.LaunchConfiguration("perception")

    sim = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            str(SHARE / "launch" / "sim.launch.py")
        ),
        launch_arguments={
            "seed": seed,
            "rooms_per_side": rooms,
            "clutter": clutter,
            "people": people,
            "rviz": rviz,
            "rviz_config": str(SHARE / "rviz" / "nav.rviz"),
        }.items(),
    )

    servers = [
        launch_ros.actions.Node(
            package=package,
            executable=executable,
            name=executable,
            output="screen",
            parameters=[PARAMS, SIM_TIME],
        )
        for package, executable in NAV2_NODES
        if executable != "map_server"  # built in _map_server
    ]

    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument("seed", default_value="0"),
            launch.actions.DeclareLaunchArgument(
                "rooms_per_side", default_value="4"
            ),
            # Off the map on purpose: only the camera can see these (R10).
            launch.actions.DeclareLaunchArgument("clutter", default_value="6"),
            launch.actions.DeclareLaunchArgument("people", default_value="0"),
            launch.actions.DeclareLaunchArgument("rviz", default_value="true"),
            # truth: the sim's exact depth (R10); model: the ONNX models
            # on the RGB image (M4.6).
            launch.actions.DeclareLaunchArgument(
                "perception", default_value="truth"
            ),
            sim,
            # ADR 0004: the sim's odom is the true pose, so map = odom.
            launch_ros.actions.Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odom",
                arguments=["--frame-id", "map", "--child-frame-id", "odom"],
            ),
            launch.actions.OpaqueFunction(function=_map_server),
            # The camera to obstacle points for the costmaps (R10, M4.6).
            launch_ros.actions.Node(
                package="mini_fsd_perception",
                executable="perception_node",
                name="perception",
                output="screen",
                parameters=[
                    SIM_TIME,
                    {"source": perception, "models_dir": str(MODELS)},
                ],
            ),
            # Same seed as the sim, so room numbers match the world.
            launch_ros.actions.Node(
                package="mini_fsd_mission",
                executable="mission_node",
                name="mission",
                output="screen",
                parameters=[
                    SIM_TIME,
                    {"seed": as_int(seed), "rooms_per_side": as_int(rooms)},
                ],
            ),
            *servers,
            launch_ros.actions.Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    SIM_TIME,
                    {
                        "autostart": True,
                        "node_names": [name for _, name in NAV2_NODES],
                    },
                ],
            ),
        ]
    )
