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

See [environment benchmarks](benchmarks/README.md) for reproducible runs and reference results.
