"""Checks nav.launch.py and the Nav2 parameters against the robot model.

Driving to a goal is a hand test (R8 in the plan). These checks catch the
mistakes that are easy to make and slow to find in RViz: a node missing from
the launch, a map that doesn't match the sim's seed, or Nav2 limits that
disagree with the robot in mjcf.py.
"""

import importlib.util
import pathlib

import launch
import launch.actions
import launch_ros.actions
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from minifsd.world import floor_plan, mjcf, occupancy

SHARE = pathlib.Path(get_package_share_directory("mini_fsd_bringup"))
NAV2_NODES = [
    "map_server",
    "planner_server",
    "controller_server",
    "behavior_server",
    "bt_navigator",
]


def _module():  # the nav.launch.py module
    path = SHARE / "launch" / "nav.launch.py"
    spec = importlib.util.spec_from_file_location("nav_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _description() -> launch.LaunchDescription:
    return _module().generate_launch_description()


def _nodes(seed: str = "0") -> list[launch_ros.actions.Node]:
    # Nodes built at launch time (inside an OpaqueFunction) need a context
    # that holds the launch arguments.
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {"seed": seed, "rooms_per_side": "2", "rviz": "false"}
    )
    nodes = []
    for entity in _description().entities:
        if isinstance(entity, launch_ros.actions.Node):
            nodes.append(entity)
        elif isinstance(entity, launch.actions.OpaqueFunction):
            nodes += [
                e
                for e in entity.execute(context) or []
                if isinstance(e, launch_ros.actions.Node)
            ]
    return nodes


def _params() -> dict:
    parsed: dict = yaml.safe_load((SHARE / "config" / "nav2.yaml").read_text())
    return parsed


def test_starts_the_nav2_servers_and_their_lifecycle_manager() -> None:
    executables = {n.node_executable for n in _nodes()}
    assert set(NAV2_NODES) <= executables
    assert "lifecycle_manager" in executables


def test_starts_the_mission_node() -> None:
    names = [(n.node_package, n.node_executable) for n in _nodes()]
    assert ("mini_fsd_mission", "mission_node") in names


def test_map_is_the_sims_floor_plan() -> None:
    yaml_path = _module().write_map(seed=7, rooms_per_side=2)
    assert yaml_path.name == "seed_7.yaml"
    pgm = yaml_path.with_suffix(".pgm").read_bytes()
    expected = occupancy.to_image(
        occupancy.from_floor_plan(
            floor_plan.generate_floor_plan(7, rooms_per_side=2)
        )
    )
    assert pgm.endswith(expected.tobytes())


def test_map_to_odom_is_a_static_identity() -> None:
    # ADR 0004: the sim's odom is the true pose, so map and odom coincide.
    (tf,) = [
        n for n in _nodes() if n.node_executable == "static_transform_publisher"
    ]
    assert tf.node_package == "tf2_ros"
    args = list(tf._Node__arguments)  # launch_ros keeps them private
    # With no --x/--yaw arguments, the transform is the identity.
    assert args == ["--frame-id", "map", "--child-frame-id", "odom"]


@pytest.mark.parametrize("costmap", ["local_costmap", "global_costmap"])
def test_costmaps_use_the_robot_radius(costmap: str) -> None:
    p = _params()[costmap][costmap]["ros__parameters"]
    assert p["robot_radius"] == pytest.approx(mjcf.ROBOT_RADIUS)
    assert p["robot_base_frame"] == "base_link"


def test_controller_stays_inside_the_sims_limits() -> None:
    p = _params()["controller_server"]["ros__parameters"]["FollowPath"]
    assert p["plugin"] == "nav2_mppi_controller::MPPIController"
    assert p["motion_model"] == "DiffDrive"
    assert abs(p["vx_max"]) <= mjcf.MAX_SPEED
    assert abs(p["vx_min"]) <= mjcf.MAX_SPEED
    assert p["wz_max"] <= mjcf.MAX_TURN


def test_planner_is_smac_2d() -> None:
    p = _params()["planner_server"]["ros__parameters"]["GridBased"]
    assert p["plugin"] == "nav2_smac_planner::SmacPlanner2D"


def test_cmd_vel_is_unstamped() -> None:
    # sim_node subscribes to geometry_msgs/Twist.
    for server in ("controller_server", "behavior_server"):
        p = _params()[server]["ros__parameters"]
        assert p["enable_stamped_cmd_vel"] is False
