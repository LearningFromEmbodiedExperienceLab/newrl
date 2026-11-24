"""Compare contact generators with batched ANYmal C and MuJoCo Warp.

``--terrain mesh`` is a slab whose top is a z=0 grid, the same surface as a
plane. ``--terrain hybrid`` adds native heightfields and box stairs beside the
slab, with worlds distributed across all three surfaces.

By default, every mode refreshes contacts each physics substep. ``--contacts
newrl`` routes supported mesh pairs through the custom kernel and delegates
other pairs to Newton.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import replace

import newton
import numpy as np
import torch
import warp as wp
from newton import JointTargetMode
from tqdm import tqdm

from newrl.configs import cs
from newrl.terrains import flat_ground_mesh
from newrl.terrain_collision import HybridTerrainCollision
from newrl.utils.profiling import StageProfiler
from walk_policy import AnymalWalkPolicy

FRAME_DT = 1.0 / 50.0
SUBSTEPS = 4
DEFAULT_JACOBIAN = "sparse"
# Body-frame command (forward m/s, lateral m/s, yaw rad/s). The turning
# radius is 0.5/0.4 = 1.25 m, so the path stays near the origin.
WALK_COMMAND = (0.5, 0.0, 0.4)


class AnymalPlaneEnv:
    """ANYmal C robots tracking a fixed turning command, one per world."""

    def __init__(
        self,
        num_envs: int = 1,
        contacts: str = "newton",
        terrain: str = "mesh",
        mesh_resolution: int = 1,
        mesh_size: float = 100.0,
        mesh_thickness: float = 1.0,
        jacobian: str = DEFAULT_JACOBIAN,
        contact_refresh: str = "substep",
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if contacts not in ("newton", "mujoco", "newrl"):
            raise ValueError(f"contacts must be 'newton', 'mujoco', or 'newrl', got {contacts!r}")
        if terrain not in ("plane", "mesh", "rough_easy", "rough_hard", "hybrid"):
            raise ValueError(f"Unknown terrain {terrain!r}")
        if contact_refresh not in ("substep", "control"):
            raise ValueError("contact_refresh must be substep or control")
        if contacts == "mujoco" and contact_refresh == "control":
            raise ValueError("MuJoCo's built-in detector runs every physics substep")
        self.device = wp.get_device()
        if not self.device.is_cuda:
            raise RuntimeError("SolverMuJoCo on this path requires CUDA.")
        self.num_envs = num_envs
        self.contacts_backend = contacts
        self.contact_refresh = contact_refresh
        self.terrain = terrain
        self.use_mujoco_contacts = contacts == "mujoco"
        self.frame_dt = FRAME_DT
        self.sim_dt = FRAME_DT / SUBSTEPS
        self.step_count = 0

        robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        policy_cfg = AnymalWalkPolicy.config()
        robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1, limit_ke=1.0e3, limit_kd=1.0e1
        )
        robot.default_shape_cfg.ke = 5.0e4
        robot.default_shape_cfg.kd = 5.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = 0.75

        # The locomotion checkpoint shipped with Newton is the Anymal C MuJoCo
        # policy. Anymal D has no policy in the repo, and the C gains and joint
        # order do not match D.
        asset_path = AnymalWalkPolicy.asset_dir()
        robot.add_usd(
            str(asset_path / "usd" / "anymal_c.usda"),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        )
        robot.joint_q[2] = 0.76
        robot.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        robot.joint_q[7:] = policy_cfg["mjw_joint_pos"]
        for i in range(len(policy_cfg["mjw_joint_stiffness"])):
            robot.joint_target_ke[i + 6] = policy_cfg["mjw_joint_stiffness"][i]
            robot.joint_target_kd[i + 6] = policy_cfg["mjw_joint_damping"][i]
            robot.joint_armature[i + 6] = policy_cfg["mjw_joint_armature"][i]
            robot.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(robot, copy_count=num_envs)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        terrain_start = builder.shape_count
        anchors = ((0.0, 0.0, 0.0),)
        self.triangle_count = 0
        if terrain == "plane":
            builder.add_ground_plane()
        if terrain in ("rough_easy", "rough_hard", "hybrid"):
            preset = cs.get("terrain", "rough_hard" if terrain == "hybrid" else terrain)
            if terrain == "hybrid":
                preset = replace(preset, rows=1)
            anchors = preset.build(builder)
        if terrain in ("mesh", "hybrid"):
            ground = flat_ground_mesh(mesh_resolution, mesh_size, mesh_thickness)
            x = -(preset.tile_size + mesh_size) / 2 if terrain == "hybrid" else 0.0
            builder.add_shape_mesh(
                body=-1, mesh=ground, label="ground_mesh",
                xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
            )
            self.triangle_count = 2 * mesh_resolution * mesh_resolution + 10
            if terrain == "hybrid":
                anchors = ((x, 0.0, 0.0), *anchors)
        for world in range(num_envs):
            start = world * len(robot.joint_q)
            for axis, offset in enumerate(anchors[world % len(anchors)]):
                builder.joint_q[start + axis] += offset

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            jacobian=jacobian,
            cone="elliptic",
            impratio=100.0,
            iterations=20,
            ls_iterations=40,
            nconmax=80,
            njmax=100,
            use_mujoco_contacts=self.use_mujoco_contacts,
        )
        self.state = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        if self.use_mujoco_contacts:
            self.collision_pipeline = None
        elif contacts == "newrl":
            self.collision_pipeline = HybridTerrainCollision(
                self.model, range(terrain_start, builder.shape_count)
            )
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = (
            self.collision_pipeline.contacts() if self.collision_pipeline is not None else None
        )
        self.graph = None
        self._viewer = None
        self.policy = AnymalWalkPolicy(self.model, num_envs, WALK_COMMAND)

    def simulate(self, profiler: StageProfiler | None = None) -> None:
        """One control step. Safe to capture: no host sync and a fixed substep count."""
        stage = nullcontext if profiler is None else profiler.stage
        for substep in range(SUBSTEPS):
            self.state.clear_forces()
            if self._viewer is not None and self.graph is None and profiler is None:
                self._viewer.apply_forces(self.state)
            if self.collision_pipeline is not None and (
                substep == 0 or self.contact_refresh == "substep"
            ):
                with stage("collision"):
                    self.collision_pipeline.collide(self.state, self.contacts)
            with stage("solver"):
                self.solver.step(self.state, self.state_next, self.control, self.contacts, self.sim_dt)
            self.state, self.state_next = self.state_next, self.state

    def capture(self, viewer=None) -> None:
        """Compile kernels once, then record ``simulate`` into a CUDA graph."""
        self.simulate()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        self._viewer = viewer

    def step(self, profiler: StageProfiler | None = None) -> None:
        """Queue one control step, using eager physics for stage profiling."""
        # The policy writes into the control buffer the graph already captured.
        self.policy.apply(self.state, self.control)
        if self.graph is not None and profiler is None:
            wp.capture_launch(self.graph)
        else:
            self.simulate(profiler)
        self.step_count += 1

    def base_pose(self) -> np.ndarray:
        body_q = self.state.body_q.numpy()
        poses = [body_q[i, :3] for i, label in enumerate(self.model.body_label) if label.endswith("/base")]
        if len(poses) != self.num_envs:
            raise RuntimeError(f"expected {self.num_envs} bases, found {len(poses)}")
        return np.asarray(poses, dtype=np.float64)


def _open_viewer(kind: str):
    """Open the window before Warp initializes CUDA.

    On this hybrid laptop the X server's GLX connection is the Intel GPU.
    Creating the pyglet context after CUDA has claimed the NVIDIA device
    fails with a GL context error.
    """
    if kind == "gl":
        viewer = newton.viewer.ViewerGL()
        for _ in range(3):
            viewer.begin_frame(0.0)
            viewer.end_frame()
        return viewer
    if kind == "rtx":
        return newton.viewer.ViewerRTX()
    raise ValueError(f"unknown viewer {kind!r}")


def _run_viewer(env: AnymalPlaneEnv, viewer, steps: int) -> None:
    """Draw until the window closes. Simulation stops after ``steps`` frames."""
    viewer.set_model(env.model)
    if env.num_envs > 1:
        viewer.set_world_offsets((2.0, 2.0, 0.0))
    sim_time = 0.0
    simulated = 0
    while viewer.is_running():
        if simulated < steps and not viewer.is_paused():
            env.step()
            simulated += 1
            sim_time += env.frame_dt
        viewer.begin_frame(sim_time)
        viewer.log_state(env.state)
        viewer.end_frame()
    viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--profile-memory", action="store_true", help="report peak Warp CUDA pool usage")
    parser.add_argument("--contacts", choices=("newton", "mujoco", "newrl"), default="newton")
    parser.add_argument(
        "--contact-refresh", choices=("substep", "control"), default="substep",
        help="test once-per-control-step refresh with an external detector",
    )
    parser.add_argument("--jacobian", choices=("sparse", "dense", "auto"), default=DEFAULT_JACOBIAN)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--viewer", choices=("none", "gl", "rtx"), default="none")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--terrain", choices=("plane", "mesh", "rough_easy", "rough_hard", "hybrid"),
        default="mesh",
    )
    parser.add_argument(
        "--profile-stages", type=int, default=0, metavar="STEPS",
        help="measure CUDA stream time in a separate eager pass",
    )
    parser.add_argument("--mesh-res", type=int, default=1, help="cells per side; 1 is two triangles")
    parser.add_argument("--mesh-size", type=float, default=20.0, help="square side length in meters")
    parser.add_argument("--mesh-thickness", type=float, default=1.0, help="slab depth below z=0, meters")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1 or args.profile_stages < 0:
        raise SystemExit("--warmup and --profile-stages must be >= 0 and --steps must be >= 1")

    viewer = _open_viewer(args.viewer) if args.viewer != "none" else None
    env = AnymalPlaneEnv(
        num_envs=args.num_envs,
        contacts=args.contacts,
        terrain=args.terrain,
        mesh_resolution=args.mesh_res,
        mesh_size=args.mesh_size,
        mesh_thickness=args.mesh_thickness,
        jacobian=args.jacobian,
        contact_refresh=args.contact_refresh,
    )
    if args.graph:
        env.capture(viewer)
    elif viewer is not None:
        env._viewer = viewer
    if viewer is not None:
        _run_viewer(env, viewer, args.steps)
        return

    for _ in tqdm(range(args.warmup), desc="Warmup"):
        env.step()
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in tqdm(range(args.steps), desc="Steps"):
        env.step()
    wp.synchronize()
    elapsed_s = time.perf_counter() - t0
    pose = env.base_pose()
    height = pose[:, 2]
    step_ms = 1.0e3 * elapsed_s / args.steps
    timing = f"step_ms={step_ms:.2f} env_steps/s={env.num_envs * args.steps / elapsed_s:.0f}"
    if args.profile_memory:
        peak = wp.get_mempool_used_mem_high(env.device)
        timing += f" warp_pool_peak_MiB={peak / 2**20:.1f}"
    if args.profile_stages:
        # Warm the eager path and time it independently of throughput.
        env.simulate()
        wp.synchronize()
        device = torch.device(str(env.device))
        profiler = StageProfiler(device)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))):
            for _ in range(args.profile_stages):
                env.step(profiler)
        for name, (stream_ms, _, calls) in profiler.totals().items():
            timing += f" eager_{name}_ms={stream_ms / args.profile_stages:.2f}"
            timing += f" {name}_calls={calls}"
    custom_pairs = sum(
        item.query_count for item in getattr(env.collision_pipeline, "mesh_colliders", ())
    )
    print(
        f"envs={env.num_envs} contacts={env.contacts_backend} "
        f"terrain={env.terrain} triangles={env.triangle_count} custom_pairs={custom_pairs} "
        f"jacobian={args.jacobian} collision_refresh={env.contact_refresh} "
        f"warmup={args.warmup} steps={args.steps} graph={args.graph} "
        f"{timing} "
        f"base_height min={height.min():.3f} mean={height.mean():.3f} max={height.max():.3f} "
        f"base_xy max_abs_x={np.abs(pose[:, 0]).max():.2f} max_abs_y={np.abs(pose[:, 1]).max():.2f}"
    )


if __name__ == "__main__":
    main()
