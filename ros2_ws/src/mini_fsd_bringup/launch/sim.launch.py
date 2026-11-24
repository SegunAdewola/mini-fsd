"""Starts the simulator and RViz.

    ros2 launch mini_fsd_bringup sim.launch.py seed:=3 rviz:=false

Teleop runs in its own terminal (`pixi run teleop`), because it reads keys
from the terminal it was started in.
"""

import pathlib

import launch
import launch.actions
import launch.conditions
import launch.substitutions
import launch_ros.actions
import launch_ros.parameter_descriptions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> launch.LaunchDescription:
    share = pathlib.Path(get_package_share_directory("mini_fsd_bringup"))
    seed = launch.substitutions.LaunchConfiguration("seed")
    rooms = launch.substitutions.LaunchConfiguration("rooms_per_side")
    rviz = launch.substitutions.LaunchConfiguration("rviz")

    def as_int(value: launch.substitutions.LaunchConfiguration) -> object:
        # Launch arguments are strings; the node declares these as integers.
        return launch_ros.parameter_descriptions.ParameterValue(
            value, value_type=int
        )

    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                "seed", default_value="0", description="Floor plan seed"
            ),
            launch.actions.DeclareLaunchArgument(
                "rooms_per_side", default_value="4"
            ),
            launch.actions.DeclareLaunchArgument(
                "rviz", default_value="true", description="Start RViz"
            ),
            launch_ros.actions.Node(
                package="mini_fsd_sim",
                executable="sim_node",
                name="sim",
                output="screen",
                # No use_sim_time here: this node publishes /clock.
                parameters=[
                    {"seed": as_int(seed), "rooms_per_side": as_int(rooms)}
                ],
            ),
            launch_ros.actions.Node(
                package="rviz2",
                executable="rviz2",
                name="rviz",
                arguments=["-d", str(share / "rviz" / "sim.rviz")],
                parameters=[{"use_sim_time": True}],
                condition=launch.conditions.IfCondition(rviz),
            ),
        ]
    )
