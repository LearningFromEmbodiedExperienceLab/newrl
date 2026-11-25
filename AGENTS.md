# NewRL development guide

NewRL is a batched robot-learning environment built on Newton, MuJoCo Warp,
and PyTorch. Newton builds and replicates the scene and provides entity I/O.
The locomotion tasks use velocity tracking. The on-policy learners are Gaussian
PPO and GenPO++.

## Compatibility

- Do not keep compatibility or legacy code. Update callers and stored formats
  to the current API instead of adding aliases, wrappers, deprecated paths, or
  fallback behavior for old interfaces.

## Algorithm implementation

- Prefer simple, self-contained implementations. Keep each algorithm, its
  configuration, and its algorithm-specific update logic together in one file.
- Keep learners and their configs independent. Use shared primitives such as
  batch utilities, GAE, and optimizers directly instead of inheriting from
  another algorithm or introducing a learner base class.
- Give each on-policy config a stable `algo_id` and a `build(env)` method.
  Register its preset under `algo` and import its module from `newrl.algos`.
- Keep single-use logic at its call site. Extract a method when it gives a
  substantial operation a clear boundary or makes that operation independently
  testable.

## Research projects

- Keep project dependencies and package sources in the project repository.
  Resolve them in that project's environment and keep `uv.lock` out of
  version control so machines can use different package indexes.
- Register project presets through a `newrl.projects` entry point with a
  `register(store, project)` callback. Use the project name as the preset
  namespace, such as `task=foo_project:AnymalLocoFlat`.
- Projects own custom training and playback scripts. Use NewRL's built-in
  commands when they fit the experiment.

## Environment lifecycle

**The environment never auto-resets.** The outer loop consumes terminal
transitions and explicitly resets the selected worlds. A step always returns
the true successor observation, including on termination or truncation; any
requested physical snapshot describes that same successor. Preserve the physics and MDP state of unselected worlds during
a partial reset.

- `StatefulEnvironment.step(sim_inputs)` accepts a TensorDict containing the
  action and returns a transition and advances the
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

## Transition schema and training

Both environment step APIs accept a TensorDict of simulator inputs containing
`action` with shape `[num_envs, action_dim]`. The environment owns the
interpretation of simulator inputs. Its step output contains the true successor
observation, MDP outputs, optional physical snapshot, reward and ending flags.

The shared collector in [rollout.py](src/newrl/algos/rollout.py) assembles a
TensorDict with batch size `[num_envs]` for each real transition.

```text
observation/<group>             decision-time observation
mdp_state/<field>               decision-time MDP output
physical_state/<entity>/<field> optional decision-time physical snapshot
action                          action applied to the simulator
<learner fields>                data produced by act()
next/
    observation/<group>         true successor observation
    mdp_state/<field>            successor MDP output
    physical_state/<entity>/<field>
                                optional true successor physical snapshot
    reward/<group>               reward for the action
    terminated                  terminal flag
    truncated                   truncation flag
    done                        terminated | truncated
    termination_terms/<name>    individual termination flags
    truncation_terms/<name>     individual truncation flags
```

- `act(decision)` adds action and learner-specific data at the root.
  The collector passes the resulting TensorDict to the environment's step,
  attaches its output at `next`, and calls `record(transition)` before
  any reset. Reset observations replace only the selected worlds in the
  carry for the following decision.
- `record(transition)` returns the row to retain for training and episode
  statistics. Stacked rollout rows have batch size `[num_envs, steps]`.
  Retain `mdp_state.episode_step` from decision time and reward groups and
  ending flags under `next`. Episode length is the decision-time count plus
  one at an ending transition.
- PPO, PPO SymAug and GenPO retain the normalized actor input as
  `policy_observation`, the behavior action log probability as
  `action_log_prob`, and critic inputs at `critic_observation` and
  `("next", "critic_observation")`. Extra algorithm-specific inputs, such
  as flow history or normalized mirrored observations, remain at the root.
  Use `next.terminated` to mask value bootstrapping and `next.done` to
  stop the advantage trace.
- SuperMimic encodes physical snapshots into `state` and
  `("next", "state")`, and copies these, decision-time replay MDP state,
  action, summed reward and ending flags into persistent replay storage.
  Its returned rollout rows retain episode statistics.
- The shared trainer uses `collect(env, steps, carry)` and
  `train_op(batch)` for all learners. A learner may implement
  `warmup(env, carry)` returning a batch and the next carry. Warmup frames
  and episode statistics count toward training totals. SuperMimic performs
  world-model and imagined actor/critic updates inside `train_op`.

## Scene, terrain, and task

- Unlike mjlab or Isaac Lab, NewRL never scales rewards by `dt`. Environment
  presets and MDP reward terms apply their configured weights directly, without
  timestep scaling. Any reward scaling is the algorithm or trainer's
  responsibility, for example scaling by the effective horizon.
- Define environment presets in [configs.py](src/newrl/configs.py), robot
  asset and actuator settings in [robots.py](src/newrl/robots.py), and
  terrain geometry in [terrains](src/newrl/terrains). The scene builds
  a template world and replicates it with Newton.
- Plane terrain is the default. Mesh terrain accepts a Newton mesh, a mesh
  file, or a mesh factory. The flat slab shares its geometry generator
  with the collision benchmark. Optional spawn locations are world-space
  ground anchors shared by the replicated worlds.
- Generated terrain builds one static tile grid and returns its spawn anchors
  during scene construction. Use native heightfields for sampled roughness
  and boxes for flat slabs and stairs. Terrain generation uses its own seed,
  while world resets sample the anchors through the MDP's per-world RNG.
- The MDP samples task state and optional reset poses. The environment
  applies the sampled poses, clears solver history, and rebuilds
  kinematics. `VelocityMDP` owns pose noise, spawn sampling, command
  scheduling, and their per-world RNG state. Carry these values in
  `mdp_state` so partial resets and replay preserve the RNG streams.
- Evaluate rewards against the command used to choose the action. Apply
  command changes after scoring and before building the next observation.
  The scene supplies projected gravity when requested by the physical read plan.
- `MDP.step(successor_physical_state, mdp_state, action)` owns task transitions
  and returns `(next_mdp_state, transition)` without mutating its inputs. Its
  shared implementation scores decision-time goals/history, advances counters,
  task memory and history, then evaluates successor endings and observations.
  Tasks customize `_advance_task_state` instead of reimplementing assembly.
  `compute_observation`, `compute_reward`, `compute_termination`, and
  `compute_truncation` are pure evaluations: never mutate state or sample RNG.
  Terms needing physical history declare `state_history` readers; the MDP
  initializes and selectively resets that history, updates it after scoring,
  and includes it in replay state. Preserve gradients through imagined history.
- Term interfaces live in [terms.py](src/newrl/mdp/terms.py). Resolve scene
  indices once in `_initialize(scene)`. Observation `shape` excludes the world dimension
  and is available after scene binding for scene-dependent terms. All batched
  scalar tensor fields have shape `[*batch, 1]`, including episode steps,
  individual rewards, episode-end flags, and task state such as RNG counters.
  Episode-end flags are boolean. Indices and other unbatched or intermediate
  tensors follow their own API requirements.
  Observation and reward groups concatenate their terms into one vector per
  group. Step outputs expose reward groups at `("reward", group_name)`.
  PPO sums the reward columns across groups before computing advantages.
- The RL environment defaults to Newton's `CollisionPipeline`.
  `task.collision=mujoco` uses MuJoCo Warp's built-in detector and requires the
  MuJoCo solver. `task.collision=newrl` routes supported primitive–terrain mesh
  pairs to the custom collider and delegates the remaining pairs to Newton.
  Every mode refreshes contacts on each physics substep. Compare modes with
  [benchmark_mesh_collision.py](benchmark_mesh_collision.py).
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
newrl play task=AnymalLocoFlat viewer=false steps=5 episode_steps=2 task.num_envs=4
```

See the [benchmark guide](benchmarks/README.md) for throughput and stage
profiling commands. Compare revisions on the same runner and workload.
Keep local histories in the ignored `benchmark-results/` directory and add
only selected shareable references under `benchmarks/reference/`. Use
anonymous labels such as `runner-01`. Benchmark exports must omit usernames,
hostnames, local paths, device identifiers, and environment variables.

MDP terms may declare `physical_fields` as a set of `(entity, field)` pairs.
The bound MDP unions these requirements across observation, reward, and episode-end
terms. Terms without a declaration retain a full scene snapshot.
`Scene._initialize` combines the MDP requirements with
`output_physical_fields` and initializes the sensors needed for those reads.
`output_physical_fields` controls which physical fields appear in outputs;
`output_mdp_fields` requests additional MDP fields. Outputs always include
`mdp_state.episode_step`, with shape `[num_envs, 1]`, describing the returned
observation: zero on reset and the successor count on step. Output snapshots
own their storage and remain valid after later steps and partial resets.
