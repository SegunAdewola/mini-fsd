# ADR 0009: Model the compute budgets with thread caps, not Docker

- Status: Accepted
- Date: 2026-11-26
- Plan step: E6

## Context

The design's "compute budget profiles" (a Must, added after the Nov 10
discussion) run the robot software in a Docker container with hard CPU and
memory caps: 6 cores and 8 GB (about an Orin NX), 4 cores and 4 GB with
perception on 2 threads (about an Orin Nano), and 2 cores and 2 GB (about a
Raspberry Pi 5).

By the end of M4 the stack runs natively under pixi (ADR 0001): MuJoCo,
RoboStack ROS 2 and Nav2, and the models in ONNX Runtime. None of it runs in
Docker. On macOS, Docker runs Linux in a VM, so the ROS packages would have to
be rebuilt for linux-aarch64, the simulator's renderer moved to the VM's
software OpenGL, and the runtime measured inside a VM. That is days of setup
with 12 days left and the real photos still to label.

Every latency number so far already runs under a thread cap: the harness sets it
for numpy, OpenCV, torch and ONNX Runtime (two threads by default), and the
perception node takes a `threads` parameter.

## Decision

The budget profiles are thread caps on the native stack: perception on 1, 2 and
4 threads, measured by the harness in one cool session (`scripts/latency.sh`)
and in the closed loop where it matters. Memory is reported as peak resident
memory, which the harness already logs. The report says the profiles are thread
caps, not containers, and that M1 cores are faster than Jetson or Pi cores.

## Consequences

- The latency tables and the closed loop measure the code that actually runs,
  with no VM in the way.
- Memory is measured, not enforced: a model that would not fit in 2 GB shows up
  as a number, not as a crash.
- Other processes (the simulator, RViz) still share the cores. Timing in a quiet
  session (T4) is what keeps that out of the numbers.

## Alternatives considered

- **Docker with `--cpus` and `--memory`, as designed.** The faithful version,
  but the setup cost above, and measurements through a VM.
- **Drop the profiles altogether.** Loses the one link from "fast model" to
  "which hardware could run it".
