"""Mini FSD core library.

All logic that does not need ROS lives here (ADR 0003): the simulated world,
geometry, perception, evaluation and data tools. ROS nodes in ros2_ws/src
only wrap these modules. This package must never import ROS.
"""

__version__ = "0.1.0"
