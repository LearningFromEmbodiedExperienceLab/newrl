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

`ContactSensorConfig` can enable `track_first_contact`, `track_air_time`, and
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
the AS2W tire visuals and uses spheres for wheel contacts. The other two
presets use a rounded wheel, modeled as a disk swept by a sphere. AS2W has
an 86 mm rolling radius and 26 mm half-width, with tread centers 54.6 mm
outboard of the foot origins. M20S uses 90 mm and 27 mm respectively.
Newton and NewRL collide the analytic barrel cylinder. MuJoCo native
collision uses a 32-sided convex mesh sampled from the same profile. Leg actions set
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
custom detector on supported mesh pairs and rounded wheels against infinite planes. The custom mode delegates heightfields,
boxes, and other pairs to Newton. All three refresh contacts on each physics
substep in training.

## Robot asset cache

See the [robot contribution guide](robots.md) for adding robot definitions
and asset recipes to NewRL or a research project.

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

## Passive objects and per-world variants

Run the tabletop example to see a G1 and a fixed four-legged table in each
world. The four default worlds include all three tabletop shapes (box, sphere
and cylinder). Steel simulates the mixed primitives, and every transition
earns one survival point.

```sh
newrl play task=G1Tabletop
```

Add named configs to `EnvironmentConfig.objects`. A box, sphere, cylinder,
or chair has one rigid body. A door exposes a revolute `hinge` and a drawer
exposes a prismatic `slide`. Both fixtures default to a fixed base.
Positions are in meters and orientations are XYZW quaternions.

```python
from newrl.configs import G1Locomotion
from newrl.assets.objects import BoxConfig, DoorConfig, ObjectVariants
from newrl.sensors import ContactSensorConfig

task = G1Locomotion(
    num_envs=32,
    objects={
        "parcel": ObjectVariants((
            BoxConfig(position=(1, 0, 0.2)),
            BoxConfig(position=(1, 0, 0.2), half_extents=(0.2, 0.1, 0.15)),
        ), weights=(3, 1)),
        "door": DoorConfig(position=(2, 0, 0)),
    },
    object_seed=17,
    sensors={"parcel_touch": ContactSensorConfig("robot", ".*wrist.*", counterpart="parcel")},
    output_physical_fields=(("door", "joint_pos"), ("parcel_touch", "loaded")),
)
```

For a rigid or articulated USD, use `USDObjectConfig(asset=path)`.
The default preserves the asset's root attachment. Set `fixed=True` or
`fixed=False` to request a fixed or floating base on import. Projects can
supply a config with a `build() -> newton.ModelBuilder` method and resolve
their own assets there. Use body and joint labels that remain stable across
variants. Scalar joint observations and reset overrides address hinges and
slides by entity order. Free-root pose and twist overrides require one free
root. Model defaults apply on reset and velocities default to zero. Object
motion participates in selective reset and stateless replay. Only robots
consume action columns. Goals and task progress belong in the MDP state.

### Heterogeneity support

The scene samples each object's variant at construction using `object_seed`.
Inspect `scene.object_variant_ids[name][world]` for its index. Resets keep
that assignment. Replay requires the same scene layout and assignment.
Without variants, Newton replicates one template in bulk. With variants,
NewRL builds each asset once, caches templates for the sampled combinations
and adds the selected template to each world.

All variants of a named object must share ordered body and joint names, joint
types, connectivity, DOF dimensions and articulation boundaries. This keeps
entity tensors, policy inputs and per-world simulator checkpoints rectangular.
Mass, density, friction, geometry dimensions and initial poses can differ.
With shape friction randomization, each world exposes its matched shapes in
model order. Shorter rows have trailing zeros in the physical snapshot.

| Backend | Geometry across worlds |
| --- | --- |
| Steel | Each world supplies its own shapes. A single-body chair assembled from six boxes can share a batch with a single box. Steel groups articulation work by size internally. NewRL currently requires matching articulation topology for its batched state API. |
| MuJoCo Warp | Corresponding primitive slots must match body, type, contact dimension and priority. Primitive sizes can vary. Newton can pad missing mesh slots. Native MuJoCo contacts also require compatible collision flags, groups and exclusions. Padding has restrictions with explicit MuJoCo contact pairs. |

For topology heterogeneity, a future design would need per-world entity
bindings with validity masks or separate batches per topology, along with
corresponding observation, action and checkpoint layouts. Backend geometry
support alone does not supply those task-level contracts.
