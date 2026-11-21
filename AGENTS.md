# NewRL development guide

NewRL is a batched robot-learning environment built on Newton, MuJoCo Warp,
and PyTorch. Newton builds and replicates the scene and provides entity I/O.
The runnable task is ANYmal C velocity tracking. Spot and AS2 asset factories
and the PPO implementation are still stubs.

## Environment lifecycle

**The environment never auto-resets.** The outer loop consumes terminal
transitions and explicitly resets the selected worlds. A step always returns
the true successor observation, including on termination or truncation; any
requested physical snapshot describes that same successor. Preserve the physics and MDP state of unselected worlds during
a partial reset.

- `StatefulEnvironment.step(action)` returns a transition and advances the
  environment's internal physics and MDP state. Use this API for play and
  ordinary rollout collection. Call `reset()` before the first step and
  `reset(env_ids)` to reset selected worlds.
- `StatelessEnvironment.step(sim_state, mdp_state, sim_inputs)` returns
  `(next_sim_state, next_mdp_state, transition)`. The simulator checkpoint
  includes persistent solver buffers as well as Newton state. The
  optional `physical_state` in the transition is a physical snapshot. Use the
  explicit simulator checkpoint for branching or replay. A partial stateless
  reset requires both prior checkpoints and `env_ids`.
- Keep both APIs on the shared transition implementation in
  [env.py](src/newrl/env.py). Preserve ordering between Torch and Warp when
  using shared GPU buffers. CUDA graph capture is opt-in and covers the
  physics loop. Preserve both starting-buffer cases for odd substep counts.

## Scene, terrain, and task

- Define environment presets in [configs.py](src/newrl/configs.py), robot
  asset and actuator settings in [robots.py](src/newrl/robots.py), and
  terrain geometry in [terrain.py](src/newrl/terrain.py). The scene builds
  a template world and replicates it with Newton.
- Plane terrain is the default. Mesh terrain accepts a Newton mesh, a mesh
  file, or a mesh factory. The flat slab shares its geometry generator
  with the collision benchmark. Optional spawn locations are world-space
  ground anchors shared by the replicated worlds.
- The MDP samples task state and optional reset poses. The environment
  applies the sampled poses, clears solver history, and rebuilds
  kinematics. `VelocityMDP` owns pose noise, spawn sampling, command
  scheduling, and their per-world RNG state. Carry these values in
  `mdp_state` so partial resets and replay preserve the RNG streams.
- Evaluate rewards against the command used to choose the action. Apply
  command changes after scoring and before building the next observation.
  The scene supplies projected gravity when requested by the physical read plan.
- Term interfaces live in [terms.py](src/newrl/mdp/terms.py). Resolve scene
  indices once in `_initialize(scene)`. Observation `shape` excludes the world dimension
  and is available after scene binding for scene-dependent terms. All batched
  scalar tensor fields have shape `[*batch, 1]`, including episode steps,
  individual rewards, episode-end flags, and task state such as RNG counters.
  Episode-end flags are boolean. Indices and other unbatched or intermediate
  tensors follow their own API requirements.
  An observation group concatenates its terms into one vector. A reward group
  keeps each named term under `reward_terms`, and `reward` is their sum.
- The RL environment uses Newton's `CollisionPipeline`. The custom terrain
  collider is exercised by [benchmark_mesh_collision.py](benchmark_mesh_collision.py).
  Solver and timing defaults are configured in `EnvironmentConfig`. MuJoCo Warp
  is the default backend; Steel uses augmented drives and point friction so
  its per-world checkpoints remain replayable. Its actuator torque is
  reconstructed from the final substep's implicit drive dynamics.
  `EnvironmentConfig.solver` is a backend-specific solver config; choose a
  registered config with `task.solver=mujoco` or `task.solver=steel`.
- Terrain contact sensors bind named robot bodies and terrain shape IDs at
  initialization. The physical read plan enables solved-force export on the
  final physics substep when needed. Sensor fields are `force_w` (net force in
  newtons) and `loaded` (force magnitude above the configured threshold).
  Reset clears selected sensor rows. Stateless checkpoints preserve reduced
  forces for unselected worlds without storing raw contact rows.
  Optional first-contact, air-time, and touchdown-position tracking samples
  every physics substep. Sensor history and control-step events are part of
  the simulator checkpoint and are reset only for selected worlds.

## Validation and benchmarking

Run the environment regression suite and the standalone terrain collision
tests after changing the relevant paths. The CUDA suite covers stateful and
stateless parity, checkpoint replay, selective resets, and mesh spawn replay.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m unittest -v test_terrain_collision
uv run python scripts/play.py task=anymal_loco_flat viewer=false steps=5 episode_steps=2 task.num_envs=4
```

See the [benchmark guide](benchmarks/README.md) for throughput and stage
profiling commands. Compare revisions on the same runner and workload.
Keep local histories in the ignored `benchmark-results/` directory and add
only selected shareable references under `benchmarks/reference/`. Use
anonymous labels such as `runner-01`. Benchmark exports must omit usernames,
hostnames, local paths, device identifiers, and environment variables.

MDP terms may declare `physical_fields` as a set of `(entity, field)` pairs.
The bound MDP unions these requirements across observation, reward, and episode-end
terms. Terms without a declaration retain a full scene snapshot. Set
`EnvironmentConfig.physical_fields` to request additional internal read fields.
`output_physical_fields` controls which physical fields appear in outputs;
`output_mdp_fields` requests additional MDP fields. Outputs always include
`mdp_state.episode_step`, with shape `[num_envs, 1]`, describing the returned
observation: zero on reset and the successor count on step. Output snapshots
own their storage and remain valid after later steps and partial resets.
