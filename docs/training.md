# Training and learner reference

## Logging, checkpoints, and distributed runs

Enable W&B with `logger.enabled=true`, or add `logger.mode=offline` to
record locally. The trainer names runs `<run-id>-<task>-<algo>-<UTC timestamp>`,
groups them by task in W&B, and stores logs under `outputs_train/<task>/wandb/`
relative to the working directory. Use `logger.name` or `logger.group` to
override the W&B display name or group.

W&B groups training metrics into three sections.

| Section | Horizontal axis | Metrics |
| --- | --- | --- |
| `learning` | `env_frames` | Episode return and length, reward, and termination and truncation rates |
| `performance` | `iteration` | Rollout, update, and iteration time in ms, throughput in environment steps/s, and sampled GPU memory in MiB |
| `diagnostics` | `iteration` | Learner metrics such as KL and losses, plus per-term rewards |

`env_frames` counts transitions across all workers, including learner warmup.
`iteration` counts completed training updates starting at one. Rollout
throughput covers collection. Training throughput includes collection, episode
statistics, and the update, excluding W&B logging and checkpoint saves.

In the Performance section, you can overlay `rollout_ms` and `update_ms`
in one panel. Use separate panels for the `*_peak_allocated_mib` series
(PyTorch allocator peaks) and the `*_device_used_mib` series (whole-device
usage at phase boundaries, including other processes). Both memory series
come from rank zero and appear only on sampled iterations. The
`diagnostics/profile` table holds stage timings, detailed memory samples,
and buffer inventories with the iteration, environment frame count, and unit.
It is logged in batches and flushed at run finish.

The trainer saves a local checkpoint every 250 completed updates under
`outputs_train/<task>/checkpoints/<run-id>/`. Set `checkpoint_interval=<updates>`
to change the cadence or `checkpoint_interval=0` to disable saves.

Select Gaussian PPO with `algo=ppo` or the GenPO++ flow policy with
`algo=genpo`. GenPO adapts the actor learning rate toward `algo.desired_kl`
using augmented KL.

For multi-GPU training with PPO, PPO SymAug, or GenPO, use one torchrun worker per GPU:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    -m newrl.cli train task=G1LocoFlat algo=ppo task.num_envs=4096
```

`task.num_envs` is per GPU. The workers collect independent rollouts,
merge running normalization statistics once per rollout, and average
gradients before clipping each minibatch. Advantage normalization is local
to each worker. GenPO adapts its actor learning rate using the average
augmented KL across workers. Rank zero logs global environment frames and saves the
shared policy and optimizer. Memory profiling reports rank zero's GPU.

An experimental torchrunx launcher is available separately:

```sh
uv sync --extra torchrunx
python scripts/train_torchrunx.py 'gpu_ids=[0,1]' task=AnymalLocoFlat algo=genpo
python scripts/train_torchrunx.py gpu_ids=all task=AnymalLocoFlat algo=ppo
```

GPU indices are relative to `CUDA_VISIBLE_DEVICES`. The launcher delegates
to the built-in trainer. Use `TORCHRUNX_LOG_DIR` to select a worker log
directory (default `outputs_train/torchrunx`).

## Compilation, profiling, and symmetry

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
Both log the active penalty in W&B as `diagnostics/actor/symmetry_loss`
and compute likelihood ratios on collected samples.

## Checkpoints from W&B

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

Select coordinates with `algo.representation=generalized` (default) or
`algo.representation=maximal`. Both support
`algo.representation.integration=direct` (default) and
`algo.representation.integration=velocity`. Velocity integration predicts
pose corrections on top of motion integrated from the old and new velocities.
Maximal coordinates predict body and joint states independently, which removes
forward kinematics from imagination and allows the two predictions to disagree.

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
observation groups. Generalized coordinates store compact root and joint state
and reconstruct bodies with differentiable Torch FK using Newton scene metadata.
That path supports one free-root tree with fixed, revolute, and prismatic
descendants. Maximal coordinates also store nonroot body poses and twists.
Both decoders expose world-frame COM twists and projected gravity to the MDP.
Tasks declare compact `replay_fields` and rebuild derived state with
`restore_replay_state`, without advancing reference time or sampling RNG.
Custom tasks must include all persistent task/term history in this contract.
The trainer accepts an optional algorithm `prepare_task` hook and saves the
prepared task in checkpoints. Imagination binds only the actor/critic observation
groups, so unused real-environment diagnostics do not impose extra representation
requirements. Every evaluated term must declare its physical dependencies;
unsupported fields fail during learner construction.
