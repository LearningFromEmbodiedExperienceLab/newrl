"""Stateful environment collision sweep with isolated GPU memory measurements."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product
from multiprocessing import get_context
from statistics import median
from time import perf_counter

import torch
import warp as wp
from tensordict import TensorDict

from newrl.configs import AnymalLocomotion, cs
from newrl.env import StatefulEnvironment
from newrl.terrains import (
    BorderConfig,
    FlatMeshTerrainConfig,
    Terrain,
    TerrainConfig,
    flat_ground_mesh,
)
from newrl.utils.profiling import StageProfiler


@dataclass(frozen=True)
class MixedCollisionTerrain(TerrainConfig):
    mesh_resolution: int = 128
    mesh_size: float = 20.0

    def build(self, builder):
        native = replace(cs.get("terrain", "rough_hard"), rows=1, border=BorderConfig(width=0))
        terrain = native.build(builder)
        x = -(native.tile_size + self.mesh_size) / 2 - 1.0
        mesh = flat_ground_mesh(self.mesh_resolution, self.mesh_size, 1.0)
        builder.add_shape_mesh(
            body=-1, mesh=mesh, label="ground_mesh",
            xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
        )
        return Terrain(spawn_locations=((x, 0.0, 0.0), *terrain.spawn_locations))


def time_steps(env, initial, inputs, count, profiler=None):
    # Identical simulator and task checkpoints before warmup and each trial.
    env.restore(*initial)
    torch.cuda.synchronize(env.device)
    env.profiler = profiler
    start = perf_counter()
    for step in range(count):
        env.step(inputs[step % len(inputs)])
    torch.cuda.synchronize(env.device)
    elapsed = (perf_counter() - start) / count
    env.profiler = None
    return elapsed


def collision_case(args, n, terrain_name, detector, graph):
    res = args.mesh_resolution or 128
    terrain = {
        "plane": "plane",
        "mesh": FlatMeshTerrainConfig(resolution=res),
        "mixed": MixedCollisionTerrain(mesh_resolution=res),
    }[terrain_name]
    cfg = AnymalLocomotion(
        num_envs=n, device=args.device, substeps=args.substeps, terrain=terrain,
        collision=detector, graph_capture=graph, contact_observation=args.contact_observation,
    )
    device = torch.device(args.device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    env = StatefulEnvironment(cfg)
    env.reset()
    robot = env.scene.entities["robot"]
    mesh_worlds = n if terrain_name == "mesh" else 0
    if terrain_name == "mixed":
        root_x = wp.to_torch(env.state.joint_q)[robot.root_q_indices[:, 0]]
        mesh_worlds = int((root_x < env.scene.spawn_locations[0, 0] / 2).sum())
    initial = env.snapshot()
    phase = torch.arange(n * env.action_dim, device=device).reshape(n, -1)
    inputs = [
        TensorDict({"action": 0.1 * (phase + 0.17 * step).sin()}, [n])
        for step in range(16)
    ]
    time_steps(env, initial, inputs, args.warmup)
    samples = [time_steps(env, initial, inputs, args.steps) for _ in range(args.repeats)]
    step_s = median(samples)
    root_z = wp.to_torch(env.state.joint_q)[robot.root_q_indices[:, 2]]
    finite = bool(torch.isfinite(wp.to_torch(env.state.body_q)).all())
    if not finite:
        raise RuntimeError(f"Nonfinite physics state in {terrain_name}/{detector}, {n} worlds")
    case = {
        "workload": {
            "num_envs": n,
            "terrain": terrain_name,
            "mesh_resolution": res if terrain_name != "plane" else None,
            "mesh_triangles": 2 * res * res + 10 if terrain_name != "plane" else 0,
            "collision": detector,
            "collision_updates": "every physics substep",
            "custom_pairs": sum(
                collider.query_count for collider in getattr(env.collision, "mesh_colliders", ())
            ),
            "solver_backend": "mujoco",
            "solver_config": asdict(cfg.solver),
            "sim_dt": cfg.sim_dt,
            "substeps": cfg.substeps,
            "deterministic": cfg.deterministic,
            "ground_stiffness": cfg.ground_stiffness,
            "ground_damping": cfg.ground_damping,
            "contact_observation": args.contact_observation,
            "graph_capture": graph,
            "seed": cfg.mdp.seed,
            "action_pattern": "0.1 * sin(flat_action_index + 0.17 * (step % 16))",
            "warmup_steps": args.warmup,
            "steps": args.steps,
            "repeats": args.repeats,
            "reset_every_steps": 0,
        },
        "throughput": {"stateful": {
            "samples_ms_per_step": [sample * 1000 for sample in samples],
            "median_ms_per_step": step_s * 1000,
            "median_env_steps_per_second": n / step_s,
        }},
        "memory": {
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "warp_pool_peak_used_bytes": wp.get_mempool_used_mem_high(env.model.device),
        },
        "diagnostics": {
            "worlds_spawned_on_mesh": mesh_worlds,
            "last_trial_root_height_m": {
                "min": float(root_z.min()), "mean": float(root_z.mean()), "max": float(root_z.max()),
            },
        },
    }
    if args.profile:
        if graph:
            del env, initial
            env = StatefulEnvironment(replace(cfg, graph_capture=False))
            env.reset()
            initial = env.snapshot()
            time_steps(env, initial, inputs, args.warmup)
        profiler = StageProfiler(device)
        time_steps(env, initial, inputs, args.steps, profiler)
        stages = {}
        for name, (stream_ms, host_ms, calls) in profiler.totals().items():
            if detector == "mujoco":
                name = {"collision": "clear_forces", "solver": "solver_including_collision"}.get(name, name)
            stages[name] = {
                "calls": calls,
                "stream_ms_per_step": stream_ms / args.steps,
                "host_wall_ms_per_step": host_ms / args.steps,
            }
        case["profile"] = {"eager_stateful": stages}
    return case


def run_collision_suite(args):
    terrains = args.terrain or ("plane", "mesh", "mixed")
    detectors = args.collision or ("newton", "mujoco", "newrl")
    graphs = (False, True) if args.graph == "both" else (args.graph == "on",)
    run = {"schema_version": 1, "benchmark": "anymal-collision", "cases": []}
    # Warp's pool high-water mark is process-wide. Give each case its own CUDA
    # process so its peak cannot inherit allocations from the previous case.
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn"), max_tasks_per_child=1,
    ) as pool:
        for n, terrain, detector, graph in product(args.num_envs, terrains, detectors, graphs):
            case = pool.submit(collision_case, args, n, terrain, detector, graph).result()
            run["cases"].append(case)
            stats = case["throughput"]["stateful"]
            memory = case["memory"]
            print(
                f"collision envs={n} terrain={terrain} detector={detector} graph={graph} "
                f"custom_pairs={case['workload']['custom_pairs']} "
                f"step_ms={stats['median_ms_per_step']:.3f} "
                f"env_steps/s={stats['median_env_steps_per_second']:.0f} "
                f"warp_pool_peak_MiB={memory['warp_pool_peak_used_bytes'] / 2**20:.1f} "
                f"torch_peak_MiB={memory['torch_peak_allocated_bytes'] / 2**20:.1f}",
                flush=True,
            )
            for stage, timings in case.get("profile", {}).get("eager_stateful", {}).items():
                print(f"  eager {stage}={timings['stream_ms_per_step']:.3f} ms/step", flush=True)
    return run
