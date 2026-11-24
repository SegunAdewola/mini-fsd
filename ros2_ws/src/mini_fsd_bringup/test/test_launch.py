"""Checks that sim.launch.py is installed and starts the sim and RViz.

Running the whole launch is a hand test (see R6 in the plan); this catches a
missing install entry or a typo in a package or executable name.
"""

import importlib.util
import pathlib

import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def _load_description():  # returns a launch.LaunchDescription
    share = pathlib.Path(get_package_share_directory("mini_fsd_bringup"))
    path = share / "launch" / "sim.launch.py"
    spec = importlib.util.spec_from_file_location("sim_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _nodes() -> list[launch_ros.actions.Node]:
    return [
        e
        for e in _load_description().entities
        if isinstance(e, launch_ros.actions.Node)
    ]


def test_starts_the_sim_node() -> None:
    names = [(n.node_package, n.node_executable) for n in _nodes()]
    assert ("mini_fsd_sim", "sim_node") in names


def test_starts_rviz_with_the_installed_config() -> None:
    share = pathlib.Path(get_package_share_directory("mini_fsd_bringup"))
    assert (share / "rviz" / "sim.rviz").is_file()
    names = [(n.node_package, n.node_executable) for n in _nodes()]
    assert ("rviz2", "rviz2") in names
