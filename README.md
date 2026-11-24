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
groups them by task in W&B, and stores logs under `outputs_train/<task>/wandb/`
relative to the working directory. Use `logger.name` or `logger.group` to
override the W&B display name or group.

The trainer saves a local checkpoint every 250 completed updates under
`outputs_train/<task>/checkpoints/<run-id>/`. Set `checkpoint_interval=<updates>`
to change the cadence or `checkpoint_interval=0` to disable saves.

Select Gaussian PPO with `algo=ppo` or the GenPO++ flow policy with
`algo=genpo`. GenPO adapts the actor learning rate toward `algo.desired_kl`
using augmented KL.

Wheeled locomotion presets are available as `task=M20SLocoFlat`,
`task=AS2WLocoFlat`, and `task=AS2WSphereLocoFlat`. The sphere variant keeps
the AS2W tire visuals and uses spheres for wheel contacts. Leg actions set
joint position offsets, and wheel actions set target angular velocities.
The actuator gains and action scales are starting values for training.

Training compiles `policy.act` by default. Set `compile_policy=false` to
run the policy eagerly.

Set `memory_profile_iters=3` to log GPU memory on the first iteration and
three iterations after the first two warmup iterations. Set
`memory_profile_warmup` to change when that window starts. The report shows
buffer inventories, PyTorch rollout and update peaks, and total GPU memory
at phase boundaries. Total GPU readings include other processes on the device.
When sampling ends, training writes a chart to
`outputs_train/<task>/profiles/<run-id>/gpu_memory.svg`.

For left–right actor regularization on the ANYmal, Spot, and AS2 flat tasks,
select `algo=ppo_symaug` or set `algo=genpo algo.symmetry_coef=1.0`.
PPO SymAug defaults to a coefficient of `1.0`. GenPO defaults to `0.0`.
Both log the active penalty as
`actor/symmetry_loss` and compute likelihood ratios on collected samples.

Play a trained on-policy learner with the task saved in its checkpoint.

```sh
uv run python scripts/play.py checkpoint=path/to/checkpoint.pt
```

Playback defaults to four worlds and deterministic actions. Use `num_envs=1`
for a single world, `deterministic=false` to sample actions, or
`viewer=false steps=100` for a headless run.
Use `uv run --extra viser python scripts/play.py task=AnymalLocoFlat viewer=viser`
to open playback in a browser. Viser listens on `127.0.0.1:8080` by default.
Set `viser_host` and `viser_port` to change the listen address.
In the browser's **Robot view** panel, choose a world and robot, then enable
**Follow robot**. Enable **Camera frustums** to see the selected robot's
camera mount and field of view. Use **Camera** if the robot has several
cameras. Active cameras show their latest RGB or depth image on the frustum.
In depth images, near surfaces appear brighter and invalid pixels are black.
Training defaults to `discard_unused_obs=true` and skips groups outside the
policy and critic inputs. Playback defaults to `discard_unused_obs=false`,
which keeps the task's observations available even with a blind checkpoint.

Projects with a static Gaussian splat PLY can use `GaussianCameraConfig`
from `newrl.sensors` with `RGBCameraObservation` or `DepthCameraObservation`.
Set `mode` to `rgb`, `depth`, or `rgbd`. Configure `world_from_cloud` to
align the cloud and install `gsrender[gsplat]` in the project environment.
Gaussian cameras default to 20 Hz. Set `frequency_hz` to change the rate or
`update_every` to specify a fixed number of control steps between frames.
Gaussian depth measures expected forward-axis depth and reports pixels below
`min_alpha` as invalid.

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

## Research projects

Install NewRL in your current environment to expose the `newrl` command.
From an existing NewRL development environment, refresh the entry point
with `uv pip install --no-deps -e .`.

```sh
newrl project create foo_project
```

With an editable NewRL checkout, this creates the project under the
`newrl-projects/` directory beside the checkout. With an installed package,
the default is `newrl-projects/` under the current directory. Use `--dir`
to choose the parent directory.

Follow the generated README to create the project's own environment and
install the package. Each project chooses its dependency sources and indexes
and can use `uv pip install` without a lockfile. NewRL's train and play
commands discover presets from projects installed in the active environment
through the `newrl.projects` entry-point group.

```sh
.venv/bin/newrl train task=foo_project:AnymalLocoFlat algo=ppo
```

Run the commands from the project directory to keep outputs and downloads
there. Projects can provide their own training and playback scripts when the
built-in commands don't fit their workflow.

For zsh completion, activate the environment whose presets you want to see
and run `scripts/install_zsh_completion.sh` from the NewRL checkout. Re-run
it after installing a project or changing its presets.

See [environment benchmarks](benchmarks/README.md) for reproducible runs and reference results.

## Procedural locomotion terrain

```sh
newrl train task=AnymalLocoFlat task.terrain=rough_easy algo=ppo
newrl play task=AnymalLocoFlat task.terrain=rough_hard
```

`rough_easy` mixes two native heightfield roughness scales. `rough_hard`
mixes rougher heightfields with box stairs and pyramid slopes. `flat_tiles` uses box slabs and
`plane` selects the infinite plane. The worlds share one static terrain grid
and sample the tile centers as spawn anchors at reset. Set
`task.terrain.seed=1` to change the generated roughness or
`task.terrain.rows=4` to repeat the tile families along another row.
Generated terrains default to outer walls. Use `task.terrain.border=apron`
for a flat border and `task.terrain.border.width=5.0` to set its width.
`task.terrain.border=wall` selects walls explicitly.

Set `task.collision=newton` for the default detector, `task.collision=mujoco`
for MuJoCo Warp's built-in detector, or `task.collision=newrl` to use the
custom detector on supported mesh pairs. The custom mode delegates heightfields,
boxes, and other pairs to Newton. All three refresh contacts on each physics
substep in training.
