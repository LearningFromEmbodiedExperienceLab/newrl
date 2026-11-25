# Simulation and task design

## Solvers, state, and contact sensors

The environment defaults to Newton's MuJoCo Warp solver. Set
`AnymalLocomotion(solver=SteelSimConfig())` to use Newton's Steel solver, or
run `newrl play task=AnymalLocoFlat task.solver=steel`. Steel uses augmented
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

In Python, use `AnymalLocomotion(contact_observation=True, foot_slip_weight=0.1)`
to add privileged foot load flags and a horizontal foot slip penalty.
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

## Randomization and wheeled robots

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
