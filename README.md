# Mini FSD

Camera-only self-driving for a small indoor robot, in simulation. You type "go
to room 103"; the robot plans a route, drives there using only its forward
camera to see obstacles, and checks the door sign when it arrives. Perception is
trained mostly on synthetic data and tested on real indoor photos. Everything
runs within an edge compute budget on an Apple M1 laptop with 8 GB of RAM.

A CS5330 (Pattern Recognition and Computer Vision) final project at Northeastern
University, inspired by Tesla's camera-only FSD.

## Status

Work in progress. Walking skeleton under construction. See the
[implementation plan](docs/implementation-plan.md) for the current step.

## Stack

- **Simulation:** MuJoCo, with procedurally generated buildings.
- **Robot software:** ROS 2 Jazzy and Nav2 (Smac planner, MPPI controller),
  installed natively on macOS through RoboStack and pixi.
- **Perception:** a model comparison for detection, floor segmentation and
  monocular depth, then a shared-backbone multi-task network, exported to ONNX
  and quantized for the edge budget.

## Quick start

Requirements: an Apple Silicon Mac with macOS 14 or later, and
[pixi](https://pixi.sh).

```shell
git clone git@github.com:SegunAdewola/mini-fsd.git
cd mini-fsd
pixi install          # ROS 2, MuJoCo and tools into .pixi/ (several GB)
pixi run test         # unit and property tests
pixi run lint         # all linters (Google style guides)
```

Running the simulator (`pixi run sim`) arrives with step R6.

## Documentation

- [Docs index](docs/README.md)
- [Design](docs/design.md)
- [Design decisions (ADRs)](docs/decisions/)
- [Study guide](docs/study-guide/README.md)

## License

Apache-2.0. See [LICENSE](LICENSE). Third-party models and datasets keep their
own licenses and are not included in this repository.
