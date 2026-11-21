"""Compare stateful stepping and stateless checkpoint round trips on CUDA."""

import argparse
import json
import re
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from benchmark_metadata import collect_metadata
from tensordict import TensorDict

from newrl.configs import AnymalLocomotion
from newrl.env import StatefulEnvironment, StatelessEnvironment
from newrl.utils.profiling import StageProfiler


def measure(
    env, actions, inputs, reset_ids, steps, reset_every, mdp=None, *, profiler=None
):
    stateless = isinstance(env, StatelessEnvironment)
    if stateless:
        sim, mdp, _ = env.reset(mdp_state=mdp)
    else:
        env.reset()
    # The initial reset is outside the measurement. Subsequent scheduled
    # partial resets are timed, including checkpoint copies on the stateless path.
    torch.cuda.synchronize(env.device)
    env.profiler = profiler
    start = perf_counter()
    for step in range(steps):
        frame = step % len(actions)
        if stateless:
            sim, mdp, _ = env.step(sim, mdp, inputs[frame])
        else:
            env.step(actions[frame])
        if reset_every and (step + 1) % reset_every == 0:
            ids = reset_ids[(step // reset_every) % len(reset_ids)]
            if stateless:
                sim, mdp, _ = env.reset(sim, mdp, env_ids=ids)
            else:
                env.reset(ids)
    torch.cuda.synchronize(env.device)
    elapsed = (perf_counter() - start) / steps
    env.profiler = None
    return elapsed, mdp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[4, 2048, 4096])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--solver-backend", choices=("mujoco", "steel"), default="mujoco"
    )
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--reset-every",
        type=int,
        default=100,
        help="reset alternating halves every N steps, or use 0 for step-only timing",
    )
    parser.add_argument("--graph", choices=("both", "on", "off"), default="both")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="report CPU wall time and CUDA stream time by stage",
    )
    parser.add_argument("--output", type=Path, help="save a new JSON results file")
    parser.add_argument(
        "--machine-id", help="optional anonymous runner label, e.g. runner-01"
    )
    args = parser.parse_args()
    if args.machine_id is not None and not re.fullmatch(
        r"runner-[0-9]{1,8}", args.machine_id
    ):
        parser.error("--machine-id must use the form runner-01")
    if args.output is not None and args.output.exists():
        parser.error("--output must name a new file")
    run = {"schema_version": 1, "benchmark": "anymal-environments", "cases": []}
    if min(*args.num_envs, args.steps, args.repeats, args.substeps) < 1:
        parser.error(
            "environment counts, steps, repeats, and substeps must be positive"
        )
    if args.warmup < 3 or args.reset_every < 0:
        parser.error(
            "warmup must be at least 3 and reset frequency must be nonnegative"
        )

    graphs = (False, True) if args.graph == "both" else (args.graph == "on",)
    for n in args.num_envs:
        for graph in graphs:
            cfg = AnymalLocomotion(
                num_envs=n,
                device=args.device,
                substeps=args.substeps,
                graph_capture=graph,
                solver_backend=args.solver_backend,
            )
            envs = (StatefulEnvironment(cfg), StatelessEnvironment(cfg))
            device = envs[0].device
            if args.output is not None and "metadata" not in run:
                run["metadata"] = collect_metadata(device, args.machine_id)
            phase = torch.arange(n * envs[0].action_dim, device=device).reshape(n, -1)
            actions = [0.1 * (phase + 0.17 * frame).sin() for frame in range(16)]
            inputs = [
                TensorDict({"action": action}, batch_size=[n]) for action in actions
            ]
            reset_ids = [
                torch.arange(start, n, 2, device=device) for start in range(min(n, 2))
            ]
            measure(envs[0], actions, inputs, reset_ids, args.warmup, args.reset_every)
            _, mdp = measure(
                envs[1], actions, inputs, reset_ids, args.warmup, args.reset_every
            )
            times = [[], []]
            for repeat in range(args.repeats):
                for index in (repeat % 2, 1 - repeat % 2):
                    elapsed, context = measure(
                        envs[index],
                        actions,
                        inputs,
                        reset_ids,
                        args.steps,
                        args.reset_every,
                        mdp if index == 1 else None,
                    )
                    times[index].append(elapsed)
                    if index == 1:
                        mdp = context
            stateful, stateless = map(median, times)
            case = {
                "workload": {
                    **{
                        name: getattr(cfg, name)
                        for name in (
                            "num_envs",
                            "solver_backend",
                            "steel_iterations",
                            "steel_articulation_constraint_max",
                            "sim_dt",
                            "substeps",
                            "jacobian",
                            "solver",
                            "integrator",
                            "cone",
                            "impratio",
                            "iterations",
                            "ls_iterations",
                            "nconmax",
                            "njmax",
                            "graph_capture",
                            "deterministic",
                            "ground_stiffness",
                            "ground_damping",
                        )
                    },
                    "terrain": "plane",
                    "collision": "newton.CollisionPipeline",
                    "collision_updates": "every physics substep",
                    "seed": cfg.mdp.seed,
                    "action_pattern": "0.1 * sin(flat_action_index + 0.17 * (step % 16))",
                    "warmup_steps": args.warmup,
                    "steps": args.steps,
                    "repeats": args.repeats,
                    "reset_every_steps": args.reset_every,
                    "reset_schedule": "alternating even and odd world indices",
                },
                "throughput": {},
            }
            for name, samples in zip(("stateful", "stateless"), times, strict=True):
                case["throughput"][name] = {
                    "samples_ms_per_step": [sample * 1000 for sample in samples],
                    "median_ms_per_step": median(samples) * 1000,
                    "median_env_steps_per_second": n / median(samples),
                }
            run["cases"].append(case)
            print(
                f"envs={n} graph={graph} substeps={args.substeps} reset_every={args.reset_every} "
                f"steps={args.steps} repeats={args.repeats} "
                f"stateful_ms={stateful * 1000:.3f} stateless_ms={stateless * 1000:.3f} "
                f"stateful_env_steps/s={n / stateful:.0f} stateless_env_steps/s={n / stateless:.0f} "
                f"stateless_overhead_pct={(stateless / stateful - 1) * 100:.1f}",
                flush=True,
            )
            if args.profile:
                case["profile"] = {}
                for name, env, context in zip(
                    ("stateful", "stateless"), envs, (None, mdp), strict=True
                ):
                    profiler = StageProfiler(device)
                    measure(
                        env,
                        actions,
                        inputs,
                        reset_ids,
                        args.steps,
                        args.reset_every,
                        context,
                        profiler=profiler,
                    )
                    stages = {}
                    case["profile"][name] = stages
                    for stage, (stream_ms, host_ms, calls) in profiler.totals().items():
                        stages[stage] = {
                            "calls": calls,
                            "stream_ms_per_step": stream_ms / args.steps,
                            "host_wall_ms_per_step": host_ms / args.steps,
                        }
                        print(
                            f"profile envs={n} graph={graph} api={name} stage={stage} "
                            f"calls={calls} stream_ms/step={stream_ms / args.steps:.3f} "
                            f"host_wall_ms/step={host_ms / args.steps:.3f}",
                            flush=True,
                        )
            del envs

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as file:
            json.dump(run, file, indent=2, allow_nan=False)
            file.write("\n")


if __name__ == "__main__":
    main()
