# Environment benchmarks

## Collision modes

Compare Newton contacts, native MuJoCo Warp contacts, and the custom hybrid
pipeline on the same ANYmal C scene.

```sh
for contacts in newton mujoco newrl; do
  python benchmark_mesh_collision.py --terrain hybrid --contacts "$contacts" \
    --num-envs 1024 --profile-stages 10
done
```

All modes use the MuJoCo Warp solver. By default, each detector refreshes
contacts every physics substep. To measure the tradeoff of reusing contacts,
run `--contact-refresh control` with either external detector.
The hybrid scene distributes worlds across a triangle mesh slab, native rough
heightfield, and box stairs. Use `--terrain mesh` for the original slab-only
benchmark, or `--terrain rough_hard` for heightfields and stairs alone. The
`custom_pairs` field reports how many candidate pairs the custom kernel owns.
A mesh-free scene with `--contacts newrl` uses Newton for all contacts.

Throughput uses the selected graph mode and excludes the optional eager
profiling pass. Native MuJoCo includes collision in its reported solver time.

## Environment stepping

Run the default ANYmal sweep at 4, 2,048, and 4,096 worlds to compare the
stateful and stateless APIs. Each run uses the plane scene and exercises both
eager stepping and CUDA graphs.

```sh
uv run python scripts/benchmark_environments.py \
  --machine-id runner-01 \
  --output benchmark-results/runner-01/run-001.json
```

Use a different anonymous runner number for each physical machine and a new
file for each run. Add `--profile` to include stage timings from a separate
instrumented pass. Throughput samples exclude that pass. Use `--reset-every 0`
to measure stepping alone.

Add `--contact-observation` to include ANYmal's terrain contact sensor and
privileged load flags. With `--graph off --profile`, `contact_sensor` measures
the reduction, while force export is timed in `solver`. With graphs enabled,
both are included in `physics_graph`.

Compare revisions on the same runner with the same workload and power mode.
Use the individual trial samples to judge run-to-run variation. Cross-machine
results show hardware scaling. Keep the full history in a separate results
branch or repository, and profiler traces in artifact storage. Commit selected
reference JSON files under [reference](reference/). A dirty revision marks a
development reference whose exact source tree may differ from the recorded commit.

JSON metadata is limited to hardware models, versions, workload parameters,
anonymous runner labels, revision hashes, and dirty flags. The exporter omits
usernames, hostnames, local paths, raw command lines, environment variables,
device UUIDs, and serial numbers. Share the JSON export. Review raw logs and
profiler traces separately before sharing them.

The `runner-01.json` file is a development reference. The solver emitted
iteration-limit warnings during the sweep. Console warning overhead is included
in the wall-clock results for affected cases.
