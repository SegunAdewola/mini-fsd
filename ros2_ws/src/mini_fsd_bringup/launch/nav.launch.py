"""Starts the simulator, Nav2 on the floor plan's map, and RViz.

    ros2 launch mini_fsd_bringup nav.launch.py seed:=3

Send a goal with RViz's "2D Goal Pose" tool. Localization is the sim's true
pose (ADR 0004): map -> odom is a static identity and there is no AMCL.
"""

import pathlib
import tempfile

import launch
import launch.actions
import launch.conditions
import launch.launch_description_sources
import launch.substitutions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

from minifsd.world import floor_plan, occupancy

SHARE = pathlib.Path(get_package_share_directory("mini_fsd_bringup"))
PARAMS = str(SHARE / "config" / "nav2.yaml")
SIM_TIME = {"use_sim_time": True}
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


def generate_launch_description() -> launch.LaunchDescription:
    seed = launch.substitutions.LaunchConfiguration("seed")
    rooms = launch.substitutions.LaunchConfiguration("rooms_per_side")
    rviz = launch.substitutions.LaunchConfiguration("rviz")

    sim = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            str(SHARE / "launch" / "sim.launch.py")
        ),
        launch_arguments={
            "seed": seed,
            "rooms_per_side": rooms,
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
            launch.actions.DeclareLaunchArgument("rviz", default_value="true"),
            sim,
            # ADR 0004: the sim's odom is the true pose, so map = odom.
            launch_ros.actions.Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odom",
                arguments=["--frame-id", "map", "--child-frame-id", "odom"],
            ),
            launch.actions.OpaqueFunction(function=_map_server),
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
