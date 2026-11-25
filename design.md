# NewRL design

NewRL was started to evaluate Newton for robot learning on complex terrain,
including nonconvex meshes and contact models for wheeled quadrupeds. Newton
builds the scene and supplies entity state. NewRL owns the task and rollout
semantics, including reset sampling, episode boundaries, and policy inputs.

The stateful environment owns live simulator and MDP state. The stateless
environment passes both checkpoints explicitly for replay and branching.
Both use the same transition implementation. Terminal observations describe
the true successor, and the caller decides which worlds to reset.

- [Simulation and task design](docs/simulation.md) covers solvers, collision,
  terrain, randomization, and sensors.
- [Training and learner reference](docs/training.md) covers learner updates,
  checkpoint formats, G1 tracking, and SuperMimic.
- [Viewer guide](docs/viewer.md) covers playback, cameras, and timeline history.
- [NewRL development guide](AGENTS.md) specifies the environment and transition
  contracts for contributors.
