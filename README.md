The environment defaults to Newton's MuJoCo Warp solver. Set
`AnymalLocomotion(solver=SteelSimConfig())` to use Newton's Steel solver, or
run `newrl play task=anymal_loco_flat task.solver=steel`. Steel uses augmented
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

Enable startup domain randomization for Unitree G1 with MuJoCo Warp or Steel
(`task.solver=steel`).

```sh
newrl train task=G1LocoFlat algo=ppo task.randomization=g1
```

The preset samples pelvis mass and inertia together, foot sliding friction,
and joint target gains once per world. Samples remain fixed through resets.
Use `task.randomization.seed` to change the samples. Request
`("robot", "body_mass_scale")`, `("robot", "shape_friction")`,
or `("robot", "joint_gain_scale")` in `output_physical_fields` to
include copies in transition outputs. Stateless checkpoints carry the samples
and restore the corresponding model parameters before solver history.

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
newrl play checkpoint=path/to/checkpoint.pt
```

Playback defaults to four worlds and deterministic actions. Use `num_envs=1`
for a single world, `deterministic=false` to sample actions, or
`viewer=false steps=100` for a headless run.
Use `uv run --extra viser newrl play task=AnymalLocoFlat viewer=viser`
to open playback in a browser. Both web viewers use port `8080` by default.
If it is unavailable, they request a free port from the operating system and
print the resulting URL. Set `viewer_port` to choose a different preferred
port. A value of `0` always requests an OS-assigned port. `viser_host` changes
Viser's listen address.
In the browser's **Robot view** panel, choose a world and robot, then enable
**Follow robot**. Enable **Camera frustums** to see the selected robot's
camera mount and field of view. Use **Camera** if the robot has several
cameras. Active cameras show their latest RGB or depth image on the frustum.
For a Gaussian camera, enable **Gaussian scene** to display its full splat cloud
in the selected world. The first enable loads and uploads the cloud to that
browser client, which can take time for large scenes.
In depth images, near surfaces appear brighter and invalid pixels are black.
Training and playback default to `discard_unused_obs=true` and skip observation
groups outside the learner's inputs. Playback without a checkpoint skips all
observation groups. Set `discard_unused_obs=false` to show simulator camera
images in Viser even when the policy does not use them. The first simulator
Gaussian camera frame can trigger a one-time gsplat CUDA build that takes
several minutes. Static Gaussian scenes can still be viewed with observation
pruning enabled.

The experimental browser viewer renders Newton meshes and static Gaussian
scenes with Three.js and Gaussian Splat Lite. Build it from a source checkout
with Node.js 22.12 or newer.

```sh
npm --prefix viewer ci
npm --prefix viewer run build
uv run newrl play task=AnymalLocoFlat viewer=browser num_envs=1
```

Open the loopback URL printed by playback. The browser interpolates poses and
smooths the selected follow target while preserving manual orbit controls.
Turn off **Follow** and click the viewport to move with **WASD**. Use **Q**
to descend and **E** to ascend. Hold **Shift** to move faster or **Ctrl** to
move more slowly. **X** also slows movement when the viewport has focus,
including for keys that Chrome reserves as Ctrl shortcuts.
Press **Space** in the viewport or click **Pause simulation** to stop
simulation stepping. The camera and recording remain active while paused.
Use `viewer_port` to select the preferred port. The **Entities** panel toggles
ground, robot visuals, collision geometry, camera frustums, and individual
Gaussian clouds. Frustums follow the selected world and remain available
when playback skips camera observations.
Gaussian clouds load the first time they are enabled. Pass
`splat_path=/path/to/scene.ply` (or `.spz`) to add a cloud already aligned to
simulation coordinates. Use Viser for live camera images on the frustums.

The **Performance** panel shows browser FPS, main thread time per frame, draw
calls, and simulation speed relative to wall time. When `nvidia-smi` is
available, it also reports device-wide utilization and memory for the host GPU
running the simulation. The browser can render on a different GPU.

Use **Record** and **Stop & Save** to capture the live viewport at up to 30 FPS.
NewRL saves the video in `recordings/` under the directory where you launched
`newrl play`. Set `recording_dir=/path/to/recordings` to choose a different
directory. The viewer uses H.264 MP4 when the browser supports it and WebM
otherwise. The saved path appears in the viewer. Keep the tab and playback
open until the save completes.

### Chrome WebGPU troubleshooting

On Linux with an NVIDIA hybrid graphics setup, check `chrome://gpu` if Chrome
reports that WebGPU is unavailable or picks the integrated GPU. If you have
the `webgpu-chrome` launcher, pass it the URL printed by playback. The
equivalent command uses a dedicated Chrome profile and forces NVIDIA offload
with Vulkan on X11. Set `VIEWER_URL` to the printed URL before running it:

```sh
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
google-chrome-stable \
  --user-data-dir="${XDG_CONFIG_HOME:-$HOME/.config}/webgpu-chrome" \
  --ozone-platform=x11 \
  --enable-unsafe-webgpu \
  --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE \
  "$VIEWER_URL"
```

Use that profile for local viewer sessions. Append `?renderer=webgl` to the
viewer URL to explicitly select WebGL2 if WebGPU is still unavailable or unstable.

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
newrl play algo=ppo "checkpoint=run:$wandb_run_path"
```

Set `wandb_run_path` to `entity/project/run_id`. Append `:250` to pin
checkpoint 250. You can omit `algo` to infer it from the checkpoint. Playback
checks W&B on each launch and reuses downloaded files from
`.cache/wandb-checkpoints/`.

## Robot asset cache

Robot bundles are fetched and cooked lazily into `.cache/robots` under the
NewRL checkout (`ROBOT_MODEL_DIR` overrides this location). Complete bundles
are reused offline, even after recipe or assetx source changes. Missing or
incomplete bundles are cooked automatically. To apply recipe changes, rebuild
the selected robots explicitly:

```sh
newrl assets rebuild spot as2w
```

Robot names are `g1`, `spot`, `as2`, `as2w`, `as2w_sphere`, and `m20s`.
Rebuilding may fetch vendor sources through assetx's separate vendor cache.

## MDP term selection

Observation and reward `body_names`/`joint_names` selectors accept a name, a
full-match regex, or a sequence mixing both. Short names match the final path
component (`pelvis` matches `robot/pelvis`); queries containing `/` match the
full label. Results follow scene order by default; `preserve_order=True` follows
query order. Unmatched or overlapping queries raise an error during scene binding.

```python
EntityField("robot", "joint_pos", joint_names=".*_knee_joint")
BodyLinearVelocity("robot", ".*_foot", frame="body")
JointLimit(2.0, joint_names="(left|right)_knee_joint")
```

`EntityField` also accepts `body_names` for body pose/twist fields. Selectors
resolve once during initialization, and observation shapes reflect the selection.

## G1 motion tracking

```sh
newrl train task=G1LAFAN1Tracking algo=ppo
newrl play task=G1LAFAN1Tracking
```

The tracking task lazily downloads the official G1-retargeted LAFAN1 CSVs from
[LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
into `.cache/lafan1` under the NewRL checkout. It retains the upstream README
and license beside the CSVs. The first run resamples the 30 Hz motions to the
task control rate (50 Hz by default), computes body poses with the cooked G1
model, and caches poses and finite-difference velocities. No mjlab installation
is required. Processing and downloads happen when the environment is built,
not when task presets are imported.

Set `LAFAN1_DATASET=/path/to/g1` to use a local directory of official CSVs,
one CSV file, or a processed NewRL NPZ. `task.mdp.dataset=/path/to/data` takes
precedence over the environment variable. Processed caches are keyed by CSV
content, model XML, frame rate, and processing version.

Resets sample a clip and a nonfinal frame, including joint and root velocities.
Reference state and per-world reset RNG are checkpointed; clip ends truncate
without crossing into another motion. The actor receives proprioception, root
velocity/height, and ten future root/joint targets. Rewards track body and root
poses and world-frame link velocities, with action-rate and soft joint-limit
penalties. Body pose rewards align reference XY/yaw to the current pelvis;
root pose rewards retain the global reference. Rewards use the decision-time
reference; endings and observations use the advanced reference. Reward weights
apply directly without timestep scaling. Algorithms or trainers own any reward
scaling, for example by the effective horizon.

Physical reset targets can supply `joint_vel` and `root_twist` alongside
`joint_pos` and `root_pose`. Omitted velocities default to zero. `root_twist`
contains world-frame COM linear velocity followed by angular velocity, matching
Newton's free-root DOFs. The tracking task converts link-origin motion
velocities to this convention before resetting.

## SuperMimic

Train the explicit generalized-coordinate world-model baseline with:

```sh
newrl train task=G1LAFAN1Tracking algo=supermimic
newrl play checkpoint=outputs_train/G1LAFAN1Tracking/checkpoints/RUN/checkpoint_001000.pt
```

The learner collects noisy actor actions into a compact real-transition replay,
fits autoregressive physical dynamics and analytic reward consistency, then
improves a deterministic actor by differentiating through imagined transitions.
Independent value heads regress detached lambda returns. The shared MDP computes
observations, rewards, reference progression, and episode endings in both real
and imagined transitions. Real terminal successors enter replay before resets;
model-training windows stop contributing after their first episode boundary.

`algo.warmup_steps`, `buffer_steps`, `wm_window`, `wm_updates`, and `wm_batch`
control real replay and model fitting. `update_every` controls real collection
between updates; `horizon` and `imagination_batch` control imagined updates.
Imagination starts must have enough reference frames remaining. Tracking uses
differentiable fall survival for policy gradients while real environments retain
hard termination thresholds; time limits retain value bootstrap.

Observation, action, and return statistics update only from real collection.
Replay and imagination read frozen statistics. Reward normalization is enabled
by default: actor/critic rewards use the standard deviation of real discounted
returns with the reference implementation's maximum-return safeguard.
`algo.normalize_reward=false` uses raw rewards. Environment rewards remain
unscaled by timestep; model fitting also uses raw rewards.

The default dynamics inputs include root-frame positions of the 14 G1 tracking
bodies, in addition to invariant generalized-coordinate features and actions.
`algo.wm_observations` configures these physical observation terms independently
of the actor and critic. They are recomputed from predicted physical states
during imagination, with gradients through FK.

Defaults match the reference explicit/simple learner: Muon for hidden matrices
and zero-decay Adam for boundary matrices, biases and normalization parameters
(`algo.optim=muon_adam`, `algo.critic_optim=muon_adam`), BF16 hidden computation
with FP32 heads/state integration (`algo.amp=bf16`), and compiled dynamics/FK.
`compile_policy` compiles collection. CUDA learner updates warm up for three
calls before capture/replay; disable these with `algo.capture_wm_update=false`
and `algo.capture_policy_critic_update=false`. `algo.compile_world_model=false`
disables model compilation; `algo.amp=none` selects FP32 throughout.
The CPU path uses eager FP32 computation without capture.

Checkpoints contain the prepared task, model parameters, normalizers, all three
optimizers, and learner counters. They support ordinary playback, including a
different world count. Replay, live simulator state, and RNG streams are not
saved, so these checkpoints do not provide exact training continuation.

This version requires plane terrain and one free-root robot tree. Nonuniform
terrain and randomized physics need explicit dynamics context before they can
be modeled correctly. Latent models and learned reward-head variants from the
reference implementation are outside this baseline.

MDP evaluation uses `compute_observation`, `compute_reward`,
`compute_termination`, and `compute_truncation`. These read supplied physical
and task state without mutation; ending methods return named boolean flags.
Action-dependent rewards require an `action` entry in the supplied task state.
`MDP.step(successor_physical_state, mdp_state, action)` performs the transition:
rewards use decision-time goals/history, then task memory advances, and endings
and observations use successor memory. Tasks customize `_advance_task_state`.
The former `advance` and `observe` method names have been removed.

SuperMimic's representation interfaces are available from
`newrl.algos.supermimic`:

```python
from newrl.algos.supermimic import SuperMimicConfig
from newrl.configs import G1MotionTracking
from newrl.env import StatefulEnvironment

algo = SuperMimicConfig()
env = StatefulEnvironment(algo.prepare_task(G1MotionTracking(num_envs=4)))
learner = algo.build(env)
transition = env.reset()
state = learner.representation.encode(transition["physical_state"])
physical = learner.representation.decode(state)
memory = env.mdp.restore_replay_state(transition["mdp_state"])
```

The representation selects physical outputs independently of actor/critic
observation groups. This version exports full body snapshots, then stores compact
root/joint state through `encode`. `decode` reconstructs body poses, COM twists,
and projected gravity with differentiable Torch FK using Newton scene metadata.
It supports one free-root tree with fixed, revolute, and prismatic descendants.
Tasks declare compact `replay_fields` and rebuild derived state with
`restore_replay_state`, without advancing reference time or sampling RNG.
Custom tasks must include all persistent task/term history in this contract.
The trainer accepts an optional algorithm `prepare_task` hook and saves the
prepared task in checkpoints. Imagination binds only the actor/critic observation
groups, so unused real-environment diagnostics do not impose extra representation
requirements. Every evaluated term must declare its physical dependencies;
unsupported fields fail during learner construction.

## Research projects

Install NewRL in your current environment to expose the `newrl` command.
From an existing NewRL development environment, refresh the entry point
with `uv pip install --no-deps -e .`.

```sh
newrl project create foo_project
newrl project install https://github.com/team/foo_project.git
newrl project install git@github.com:team/foo_project.git
newrl project list
newrl project remove foo_project
```

With an editable NewRL checkout, this creates the project under the
`newrl-projects/` directory beside the checkout. With an installed package,
the default is `newrl-projects/` under the current directory. Use `--dir`
to choose the parent directory.

`project install` clones the repository, creates its own Python 3.13 `.venv`,
and installs the project with `uv pip install -e .`. The repository must declare
a `newrl.projects` entry point. Use its `.venv/bin/newrl` for train and play.
If environment setup fails, the checkout is retained so its dependency sources
can be adjusted before retrying setup.
`project list` shows project checkouts under the selected parent.
`project remove` prompts before deleting a checkout, including its environment,
assets, and uncommitted work. Pass `--yes` to confirm in scripts.
All project commands accept `--dir` to select their parent directory.

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
