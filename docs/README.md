# Documentation

Start with the design, then the implementation plan for the current step.

## Project

- [Design](design.md): what Mini FSD does, the architecture, the experiments,
  and the evaluation plan. Approved Nov 10, 2026.
- [Implementation plan](implementation-plan.md): build order, current step,
  testing ladder, open follow-ups and the decisions log.
- [Repository layout](repo.md): where code and docs live.
- [Proposal](proposal/): the Nov 10 proposal PDF and pitch notes.

## Decisions

Architecture decision records (ADRs). Each one gives the context, the decision,
its consequences and the alternatives that were rejected.

- [0000 Template](decisions/0000-template.md)
- [0001 Install ROS 2, MuJoCo and tools with pixi and RoboStack](decisions/0001-pixi-robostack-environment.md)
- [0002 Model the robot as a planar base, not a wheeled one](decisions/0002-planar-base.md)
- [0003 Keep all ROS-free logic in one Python package](decisions/0003-functional-core-ros-shell.md)
- [0004 Use the simulator's true pose instead of localization](decisions/0004-ground-truth-pose.md)
- [0005 Follow Google style guides, enforced by linters](decisions/0005-google-style-toolchain.md)
- [0006 License the repo under Apache-2.0](decisions/0006-apache-2-license.md)

## Learning

- [Study guide](study-guide/README.md): a minimum track of outside material
  (about 6 hours), a week-by-week learning path, and project reference notes.
- [Paper reading list](papers.md): papers behind every model and method we
  compare.
