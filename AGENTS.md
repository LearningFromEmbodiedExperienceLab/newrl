# NewRL development guide

NewRL is a batched robot-learning environment built on Newton, MuJoCo Warp,
and PyTorch. Newton builds and replicates the scene and provides entity I/O.
The runnable task is ANYmal C velocity tracking. Spot and AS2 asset factories
and the PPO implementation are still stubs.

## Environment lifecycle

**The environment never auto-resets.** The outer loop consumes terminal
transitions and explicitly resets the selected worlds. A step always returns
the true successor physical state and observation, including on termination
or truncation. Preserve the physics and MDP state of unselected worlds during
a partial reset.

- `StatefulEnvironment.step(action)` returns a transition and advances the
  environment's internal physics and MDP state. Use this API for play and
  ordinary rollout collection. Call `reset()` before the first step and
  `reset(env_ids)` to reset selected worlds.
- `StatelessEnvironment.step(sim_state, mdp_state, sim_inputs)` returns
  `(next_sim_state, next_mdp_state, transition)`. The simulator checkpoint
  includes persistent solver buffers as well as Newton state. The
  `physical_state` in the transition is an observation snapshot. Use the
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
  and is available after scene binding for scene-dependent terms. Rewards
  and episode-end flags have shape `[num_envs, 1]`, with boolean flags.
  An observation group concatenates its terms into one vector. A reward group
  keeps each named term under `reward_terms`, and `reward` is their sum.
- The RL environment uses Newton's `CollisionPipeline`. The custom terrain
  collider is exercised by [benchmark_mesh_collision.py](benchmark_mesh_collision.py).
  Solver and timing defaults are configured in `EnvironmentConfig`.

## Validation and benchmarking

Run the environment regression suite and the standalone terrain collision
tests after changing the relevant paths. The CUDA suite covers stateful and
stateless parity, checkpoint replay, selective resets, and mesh spawn replay.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m unittest -v test_terrain_collision
uv run python scripts/play.py --no-viewer --steps 5 --episode-steps 2
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
`EnvironmentConfig.physical_fields` to request additional transition fields.
Read-plan snapshots own their storage and remain valid after later steps.
