"""Build script for the mini_fsd_sim ROS package (ament_python)."""

from setuptools import find_packages, setup

PACKAGE = "mini_fsd_sim"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # The empty marker file that lets `ros2 pkg list` find the package.
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    # colcon test runs pytest only if it is listed here.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Oluwasegun Adewola",
    maintainer_email="51415200+SegunAdewola@users.noreply.github.com",
    description="MuJoCo simulator node: clock, odometry, TF and camera.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [f"sim_node = {PACKAGE}.sim_node:main"],
    },
)
