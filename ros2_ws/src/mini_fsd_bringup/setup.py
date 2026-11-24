"""Build script for the mini_fsd_bringup ROS package (ament_python)."""

import glob

from setuptools import setup

PACKAGE = "mini_fsd_bringup"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[],  # no Python modules, only launch and config files
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        # `ros2 launch` and the tests find these under share/<package>/.
        (f"share/{PACKAGE}/launch", glob.glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/rviz", glob.glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    # colcon test runs pytest only if it is listed here.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Oluwasegun Adewola",
    maintainer_email="51415200+SegunAdewola@users.noreply.github.com",
    description="Launch files and RViz config for Mini FSD.",
    license="Apache-2.0",
)
