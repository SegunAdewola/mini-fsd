"""Runs the closed-loop experiment headless (M4.8).

    pixi run python tools/closed_loop.py
    pixi run python tools/closed_loop.py --seeds 0 --rooms 1   # a dry run

For each perception source (truth, model) and seed, this starts
nav.launch.py without RViz, waits for Nav2 to come up, sends the seed's
missions one after another, and appends one record per mission to
runs/closed_loop/results.jsonl. The robot starts each mission where the last
one ended. It prints the table at the end; minifsd.eval.closed_loop holds
the record format, the bump counting and the table.

A mission ends when the mission node reports succeeded or failed, or after
TIMEOUT seconds of sim time. Leave the Mac alone while it runs: model mode
is CPU-bound, and other work slows the robot's perception.
"""

import argparse
import os
import pathlib
import signal
import subprocess
import time
from collections.abc import Sequence

import rclpy
import rclpy.node
from lifecycle_msgs.srv import GetState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String, UInt32

from minifsd.eval import closed_loop
from minifsd.world import floor_plan

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "closed_loop" / "results.jsonl"
CONDITIONS = ("truth", "model")
ROOMS_PER_SIDE = 4  # nav.launch.py's default, passed so the two can't drift
CLUTTER = 6
PEOPLE = 2
TIMEOUT = 120.0  # sim seconds per mission
NAV2_WAIT = 120.0  # wall seconds for Nav2 to come up
WALL_LIMIT = 600.0  # wall seconds per mission, if the sim itself stalls


class Watcher(rclpy.node.Node):
    """Listens to what a mission needs: status, sim time and bumps."""

    def __init__(self) -> None:
        """Subscribes to the mission status, /clock and the bump count."""
        super().__init__("closed_loop")
        self.status: list[str] = []
        self.sim_time = 0.0
        self.bumps = 0
        self.create_subscription(
            String, "mission/status", lambda m: self.status.append(m.data), 10
        )
        self.create_subscription(Clock, "clock", self._on_clock, 10)
        self.create_subscription(UInt32, "sim/collisions", self._on_bumps, 10)
        self.mission = self.create_publisher(String, "mission", 10)
        self._state = self.create_client(GetState, "bt_navigator/get_state")

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_bumps(self, msg: UInt32) -> None:
        self.bumps = msg.data

    def spin_for(self, seconds: float) -> None:
        """Handles messages for a while, in wall-clock seconds."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def nav2_active(self) -> bool:
        """True once bt_navigator reports the lifecycle state 'active'."""
        if not self._state.service_is_ready():
            return False
        future = self._state.call_async(GetState.Request())
        end = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        result = future.result() if future.done() else None
        return result is not None and result.current_state.label == "active"


def run_mission(w: Watcher, condition: str, seed: int, room: int) -> None:
    """Sends one mission and records how it ended."""
    w.status.clear()
    start_sim, start_bumps = w.sim_time, w.bumps
    w.mission.publish(String(data=f"go to room {room}"))
    wall_end = time.monotonic() + WALL_LIMIT
    outcome = "timeout"
    while w.sim_time - start_sim < TIMEOUT and time.monotonic() < wall_end:
        rclpy.spin_once(w, timeout_sec=0.05)
        ended = closed_loop.outcome_of(w.status, room)
        if ended is not None:
            outcome = ended
            break
    mission = closed_loop.Mission(
        condition=condition,
        seed=seed,
        room=room,
        outcome=outcome,
        seconds=round(w.sim_time - start_sim, 2),
        collisions=w.bumps - start_bumps,
    )
    closed_loop.append(mission, OUT)
    print(
        f"  room {room}: {outcome} in {mission.seconds:.0f} s, "
        f"{mission.collisions} collisions",
        flush=True,
    )


def run_launch(condition: str, seed: int, rooms: Sequence[int]) -> None:
    """One nav.launch.py run: start, wait for Nav2, missions, stop."""
    launch = (
        "source ros2_ws/install/setup.bash && ros2 launch mini_fsd_bringup "
        f"nav.launch.py seed:={seed} rooms_per_side:={ROOMS_PER_SIDE} "
        f"clutter:={CLUTTER} people:={PEOPLE} perception:={condition} "
        "rviz:=false"
    )
    log = (OUT.parent / f"launch-{condition}-{seed}.log").open("w")
    proc = subprocess.Popen(
        ["bash", "-c", launch],
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # its own process group, to stop it whole
    )
    w = Watcher()
    try:
        deadline = time.monotonic() + NAV2_WAIT
        while not w.nav2_active():
            if time.monotonic() > deadline or proc.poll() is not None:
                raise RuntimeError(f"Nav2 didn't come up; see {log.name}")
            w.spin_for(1.0)
        w.spin_for(5.0)  # let the costmaps fill from the first frames
        for room in rooms:
            run_mission(w, condition, seed, room)
    finally:
        w.destroy_node()
        # Ctrl-C to ros2 launch only; it passes one on to each node. Sending
        # it to the whole group too would interrupt each node's cleanup.
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Runs every condition and seed, then prints the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--rooms", type=int, default=4, help="missions per seed"
    )
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS
    )
    args = parser.parse_args(argv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pixi", "run", "build"], cwd=REPO, check=True)
    rclpy.init()
    try:
        for seed in args.seeds:
            plan = floor_plan.generate_floor_plan(seed, ROOMS_PER_SIDE)
            rooms = closed_loop.pick_rooms(plan, args.rooms, seed)
            for condition in args.conditions:
                print(f"{condition}, seed {seed}, rooms {rooms}", flush=True)
                run_launch(condition, seed, rooms)
    finally:
        rclpy.shutdown()
    print(closed_loop.markdown(closed_loop.summarize(closed_loop.load(OUT))))


if __name__ == "__main__":
    main()
