The environment defaults to Newton's MuJoCo Warp solver. Set
`AnymalLocomotion(solver_backend="steel")` to use Newton's Steel solver, or
run `uv run python scripts/play.py --solver-backend steel`. Steel uses augmented
position drives, point friction and an updated mass matrix each physics step.
Its iteration count and per-articulation constraint capacity are configurable
with `steel_iterations` and `steel_articulation_constraint_max`. Newton warns
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

See [environment benchmarks](benchmarks/README.md) for reproducible runs and reference results.
