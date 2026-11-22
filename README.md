The environment defaults to Newton's MuJoCo Warp solver. Set
`AnymalLocomotion(solver=SteelSimConfig())` to use Newton's Steel solver, or
run `uv run python scripts/play.py task=anymal_loco_flat task.solver=steel`. Steel uses augmented
position drives, point friction and an updated mass matrix each physics step.
Its iteration count and per-articulation constraint capacity are configurable
with `iterations` and `articulation_constraint_max` in `SteelSimConfig`. Newton warns
if the constraint capacity is exceeded.

When joint torque is requested, the Steel backend reconstructs the actuator
force from the final physics substep. It includes the implicit drive correction;
Steel clamps the explicit drive component before that correction, so the final
force can exceed the configured effort limit. It excludes passive damping and
constraint forces. MuJoCo and Steel have different dynamics and contact models;
their trajectories and actuator forces need not match.

Stateful `reset()` and `step()` return observations and a projected `mdp_state`
containing `episode_step` (shape `[num_envs, 1]`) by default. Step also returns
rewards and episode-end flags. The count belongs to the returned observation:
it is zero after reset and advances even on a terminal step. Reset explicitly
after consuming a terminal transition. Use `output_mdp_fields` and
`output_physical_fields` in the environment configuration to include additional
snapshots. The internal physical read plan also includes fields required by MDP
terms. The stateless API additionally returns complete simulator and MDP
checkpoints for replay or branching.

Use `task=anymal_loco_flat_contact` to add privileged foot load flags and a
horizontal foot slip penalty to ANYmal locomotion. In Python, use
`AnymalLocomotion(contact_observation=True, foot_slip_weight=0.1)`.
For diagnostics, request `("feet_ground", "force_w")` or
`("feet_ground", "loaded")` in `output_physical_fields`. These fields report
solved terrain force in world coordinates on the final physics substep.
Immediately after reset, selected worlds report zero force until they step.

`TerrainContactConfig` can enable `track_first_contact`, `track_air_time`, and
`track_contact_position`. The sensor then samples every physics substep. A
first-contact event records any unloaded-to-loaded transition during the
control step, including contact on the first substep after reset. Air time
accumulates while unloaded; the last completed duration is retained at
touchdown. The last touchdown position is a force-weighted world-space contact
midpoint, accompanied by `last_contact_pos_valid`, which is false until a
touchdown has been observed since reset. These histories are checkpointed and
cleared for selected worlds on reset.

`FeetAirTime` rewards each foot whose current air time lies strictly between
its configured thresholds, following mjlab's velocity task without command
gating. `AnymalLocomotion(feet_air_time_weight=...)` enables the term and
air-time tracking.

Enable W&B with `logger.enabled=true`, or add `logger.mode=offline` to
record locally. The trainer names runs `<run-id>-<task>-<algo>-<UTC timestamp>`,
groups them by task in W&B, and stores logs under the repository's
`outputs_train/<task>/wandb/` directory. Use `logger.name` or `logger.group`
to override the W&B display name or group.

The trainer saves a local checkpoint every 250 completed updates under
`outputs_train/<task>/checkpoints/<run-id>/`. Set `checkpoint_interval=<updates>`
to change the cadence or `checkpoint_interval=0` to disable saves.

Select Gaussian PPO with `algo=ppo` or the GenPO++ flow policy with
`algo=genpo`. GenPO adapts the actor learning rate toward `algo.desired_kl`
using augmented KL.

Play a trained PPO or GenPO++ policy with the task saved in its checkpoint.

```sh
uv run python scripts/play.py checkpoint=path/to/checkpoint.pt
```

Playback defaults to four worlds and deterministic actions. Use `num_envs=1`
for a single world, `deterministic=false` to sample actions, or
`viewer=false steps=100` for a headless run.

With online W&B logging enabled (`logger.enabled=true`), the trainer queues
each checkpoint for upload as soon as it is saved. On another machine with
access to the run, pull the latest uploaded checkpoint at playback startup.

```sh
uv run python scripts/play.py algo=ppo "checkpoint=run:$wandb_run_path"
```

Set `wandb_run_path` to `entity/project/run_id`. Append `:250` to pin
checkpoint 250. You can omit `algo` to infer it from the checkpoint. Playback
checks W&B on each launch and reuses downloaded files from
`.cache/wandb-checkpoints/`.

See [environment benchmarks](benchmarks/README.md) for reproducible runs and reference results.
