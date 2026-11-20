"""Batched Newton environment with explicit reset and true successor states."""

from contextlib import nullcontext

import newton
import torch
import warp as wp
from tensordict import TensorDict

from .configs import EnvironmentConfig
from .mdp.base import MDP
from .scene import Scene
from .utils.profiling import StageProfiler


class _EnvironmentCore:
    """Shared physics and task transition implementation."""

    def __init__(self, config: EnvironmentConfig, mdp: MDP | None = None):
        if config.num_envs < 1 or config.substeps < 1 or config.sim_dt <= 0:
            raise ValueError("num_envs, substeps, and sim_dt must be positive")
        if not config.robots:
            raise ValueError("Configure at least one robot")
        if not wp.get_device(config.device).is_cuda:
            raise ValueError("The batched MuJoCo Warp environment requires CUDA")
        self.config = config
        self.num_envs = config.num_envs
        self.scene = Scene(config)
        self.mdp = (mdp if mdp is not None else config.mdp)._initialize(self.scene)
        self.model = self.scene.model
        self.device = torch.device(wp.device_to_torch(self.model.device))
        self.sim = newton.solvers.SolverMuJoCo(
            self.model,
            jacobian=config.jacobian,
            solver=config.solver,
            integrator=config.integrator,
            cone=config.cone,
            impratio=config.impratio,
            iterations=config.iterations,
            ls_iterations=config.ls_iterations,
            nconmax=config.nconmax,
            njmax=config.njmax,
            use_mujoco_contacts=False,
            enable_sleeping=False,
            deterministic=wp.DeterministicMode.RUN_TO_RUN
            if config.deterministic
            else None,
        )
        self._solver_reset_defaults = {
            name: wp.to_torch(array).clone()
            for name in ("history", "mocap_pos", "mocap_quat")
            if (array := getattr(self.sim.mjw_data, name)).size
        }
        self.collision = newton.CollisionPipeline(
            self.model, deterministic=config.deterministic
        )
        self.contacts = self.collision.contacts()
        self.state = self.model.state()
        self.state_next = self.model.state()
        self._physics_graphs = {}
        self._physics_warmed_up = False
        self.control = self.model.control()
        self.action_sizes = [
            entity.q.shape[1] for entity in self.scene.entities.values()
        ]
        self.action_dim = sum(self.action_sizes)
        self.control_dt = config.sim_dt * config.substeps
        self.profiler: StageProfiler | None = None

    def _stream(self):
        return wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
        )

    def _profile_stage(self, name):
        return nullcontext() if self.profiler is None else self.profiler.stage(name)

    def _simulate(self, state, state_next, *, profile=False):
        for _ in range(self.config.substeps):
            with self._profile_stage("collision") if profile else nullcontext():
                state.clear_forces()
                self.collision.collide(state, self.contacts)
            with self._profile_stage("solver") if profile else nullcontext():
                self.sim.step(
                    state, state_next, self.control, self.contacts, self.config.sim_dt
                )
            state, state_next = state_next, state
        return state, state_next

    def _advance(self):
        if not self.config.graph_capture or not self._physics_warmed_up:
            self.state, self.state_next = self._simulate(
                self.state, self.state_next, profile=self.profiler is not None
            )
            self._physics_warmed_up = True
            return

        # An odd substep count alternates which buffer the graph starts from.
        key = (id(self.state), self.config.substeps, self.config.sim_dt)
        if key not in self._physics_graphs:
            # CUDA capture requires a non-default stream. Replay uses the caller's stream.
            stream = wp.Stream(self.model.device)
            with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream) as capture:
                self._simulate(self.state, self.state_next)
            self._physics_graphs[key] = capture.graph
        with self._profile_stage("physics_graph"):
            wp.capture_launch(self._physics_graphs[key])
        if self.config.substeps % 2:
            self.state, self.state_next = self.state_next, self.state

    def _transition(self, physical_state, result):
        result["physical_state"] = physical_state
        obs = result["observation"]
        for group, keys in (
            ("policy_observation", self.config.policy_observation_keys),
            ("critic_observation", self.config.critic_observation_keys),
        ):
            if keys:
                result[group] = torch.cat(
                    [obs[key].reshape(self.num_envs, -1) for key in keys], dim=-1
                )
        return result

    def _reset(self, env_ids, mdp_state):
        if env_ids is not None and mdp_state is None:
            raise ValueError("A selective reset requires the previous MDP state")
        ids = (
            slice(None)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        if mdp_state is None:
            mdp_state = self.mdp.initial_state(
                self.num_envs, self.action_dim, self.device
            )
        next_mdp_state, poses = self.mdp.reset(mdp_state, ids)
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[:-1][ids] = True
        self.sim.reset(self.state, world_mask=wp.from_torch(mask, dtype=wp.bool))
        wp.to_torch(self.sim.mjw_data.time)[ids] = 0
        for name, defaults in self._solver_reset_defaults.items():
            wp.to_torch(getattr(self.sim.mjw_data, name))[ids] = defaults[ids]
        joint_q = wp.to_torch(self.state.joint_q)
        for name, pose in poses.items():
            entity = self.scene.entities[name]
            if "joint_pos" in pose:
                joint_q[entity.q[ids]] = pose["joint_pos"]
            if "root_pose" in pose:
                if entity.root_q is None:
                    raise ValueError(
                        f"Entity {name!r} requires one free root for root pose resets"
                    )
                joint_q[entity.root_q[ids]] = pose["root_pose"]
        joint_mask = mask[wp.to_torch(self.model.joint_world).long()]
        newton.eval_fk(
            self.model,
            self.state.joint_q,
            self.state.joint_qd,
            self.state,
            joint_mask=wp.from_torch(joint_mask, dtype=wp.bool),
        )
        targets = wp.to_torch(self.control.joint_target_q)
        for entity in self.scene.entities.values():
            targets[entity.targets[ids]] = entity.default_position[ids]
        physical_state = self.scene.read(self.state)
        result = TensorDict(
            {"observation": self.mdp.observe(physical_state, next_mdp_state)},
            batch_size=[self.num_envs],
            device=self.device,
        )
        return next_mdp_state, self._transition(physical_state, result)

    def _step(self, action, mdp_state):
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(action.shape)}"
            )
        with self._profile_stage("action"):
            action = action.to(device=self.device, dtype=torch.float32).clone()
            for entity, values in zip(
                self.scene.entities.values(),
                action.split(self.action_sizes, dim=-1),
                strict=True,
            ):
                entity.write_action(self.control, values)
        self._advance()
        with self._profile_stage("state_read"):
            physical_state = self.scene.read(self.state)
        with self._profile_stage("mdp"):
            next_mdp_state, result = self.mdp.advance(physical_state, mdp_state, action)
            transition = self._transition(physical_state, result)
        return next_mdp_state, transition


class StatefulEnvironment(_EnvironmentCore):
    """Own live physics buffers and MDP state. Resets are always explicit."""

    def __init__(self, config: EnvironmentConfig, mdp: MDP | None = None):
        super().__init__(config, mdp)
        self.mdp_state: TensorDict | None = None

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> TensorDict:
        with self._stream(), self._profile_stage("reset"):
            self.mdp_state, transition = self._reset(env_ids, self.mdp_state)
            return transition

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> TensorDict:
        if self.mdp_state is None:
            raise RuntimeError("Call reset() before the first step")
        with self._stream():
            self.mdp_state, transition = self._step(action, self.mdp_state)
            return transition


class StatelessEnvironment(_EnvironmentCore):
    """Use explicit checkpoints and private Newton working buffers.

    A checkpoint is tied to the environment's scene and batch layout. It carries
    Newton's joint and body states and the persistent MuJoCo Warp integration
    buffers. Collision and constraint work is rebuilt each step. Custom MDP
    terms must derive their results from the supplied physical snapshot and MDP state.
    """

    def __init__(self, config: EnvironmentConfig, mdp: MDP | None = None):
        super().__init__(config, mdp)
        if self.model.particle_count:
            raise ValueError("Simulator checkpoints currently support rigid scenes")

    def _checkpoint_views(self):
        arrays = {
            name: getattr(self.state, name)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd")
        }
        arrays.update(
            {
                name: getattr(self.sim.mjw_data, name)
                for name in (
                    "qacc_warmstart",
                    "act",
                    "time",
                    "history",
                    "mocap_pos",
                    "mocap_quat",
                )
                if getattr(self.sim.mjw_data, name).size
            }
        )
        return {
            name: wp.to_torch(array).reshape(self.num_envs, -1)
            for name, array in arrays.items()
        }

    def _snapshot(self):
        return TensorDict(
            {name: view.clone() for name, view in self._checkpoint_views().items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _restore(self, sim_state):
        views = self._checkpoint_views()
        if sim_state.batch_size != torch.Size([self.num_envs]) or set(
            sim_state.keys()
        ) != set(views):
            raise ValueError("Simulator checkpoint does not match the environment")
        for name, view in views.items():
            value = sim_state[name]
            if (
                value.shape != view.shape
                or value.dtype != view.dtype
                or value.device != view.device
            ):
                raise ValueError(f"Invalid simulator checkpoint field {name!r}")
        for name, view in views.items():
            view.copy_(sim_state[name])

    @torch.no_grad()
    def reset(
        self,
        sim_state: TensorDict | None = None,
        mdp_state: TensorDict | None = None,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[TensorDict, TensorDict, TensorDict]:
        if env_ids is not None and (sim_state is None or mdp_state is None):
            raise ValueError("A selective reset requires both input checkpoints")
        with self._stream():
            if env_ids is not None:
                with self._profile_stage("checkpoint_restore"):
                    self._restore(sim_state)
            with self._profile_stage("reset"):
                next_mdp_state, transition = self._reset(env_ids, mdp_state)
            with self._profile_stage("checkpoint_snapshot"):
                snapshot = self._snapshot()
            return snapshot, next_mdp_state, transition

    @torch.no_grad()
    def step(
        self, sim_state: TensorDict, mdp_state: TensorDict, sim_inputs: TensorDict
    ) -> tuple[TensorDict, TensorDict, TensorDict]:
        with self._stream():
            with self._profile_stage("checkpoint_restore"):
                self._restore(sim_state)
            next_mdp_state, transition = self._step(sim_inputs["action"], mdp_state)
            with self._profile_stage("checkpoint_snapshot"):
                snapshot = self._snapshot()
            return snapshot, next_mdp_state, transition
