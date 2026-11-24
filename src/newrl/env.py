"""Batched Newton environment with explicit reset and true successor states."""

from contextlib import nullcontext, AbstractContextManager

import newton
import torch
import warp as wp
from tensordict import TensorDict

from .configs import EnvironmentConfig
from .mdp.base import MDP
from .scene import Scene
from .sensors import FrameCamera, TerrainContact
from .solver_adapters import MujocoSimConfig
from .terrain_collision import HybridTerrainCollision
from .utils.profiling import StageProfiler
from .utils.symmetry import SymmetryTransform


class _EnvironmentCore:
    """Shared physics and task transition implementation."""

    def __init__(self, config: EnvironmentConfig, mdp: MDP | None  =  None) -> None:
        if config.num_envs < 1 or config.substeps < 1 or config.sim_dt <= 0:
            raise ValueError("num_envs, substeps, and sim_dt must be positive")
        if not config.robots:
            raise ValueError("Configure at least one robot")
        if config.collision == "mujoco" and not isinstance(config.solver, MujocoSimConfig):
            raise ValueError("collision=mujoco requires the MuJoCo solver")
        if not wp.get_device(config.device).is_cuda:
            raise ValueError("The batched environment requires CUDA")
        self.config = config
        self.num_envs = config.num_envs
        self.scene = Scene(config)
        self.mdp = (mdp if mdp is not None else config.mdp)._initialize(self.scene)
        self.scene.resolve_physical_fields(config.physical_fields)
        self.output_physical_plan = self.scene.resolve_physical_fields(
            config.output_physical_fields
        )
        self.output_mdp_fields = tuple(
            dict.fromkeys(("episode_step", *config.output_mdp_fields))
        )
        required = self.mdp.physical_fields
        requested = (
            None
            if required is None
            else required
            | frozenset(config.physical_fields)
            | frozenset(config.output_physical_fields)
        )
        read_plan = self.scene.physical_read_plan = self.scene.resolve_physical_fields(
            requested
        )
        self.model = self.scene.model
        self.device = torch.device(wp.device_to_torch(self.model.device))
        if config.collision == "newrl":
            self.collision = HybridTerrainCollision(
                self.model, self.scene.terrain_shapes, deterministic=config.deterministic
            )
        elif config.collision == "newton":
            self.collision = newton.CollisionPipeline(
                self.model, deterministic=config.deterministic
            )
        else:
            self.collision = None
        contacts = self.collision.contacts() if self.collision is not None else None
        needs_torque = any(
            "joint_torque" in fields for fields in read_plan.values()
        )
        active_sensors = {
            name: sensor
            for name, sensor in self.scene.sensors.items()
            if name in read_plan
        }
        for sensor in active_sensors.values():
            sensor.initialize(self.scene)
        self._contact_sensors = {
            name: sensor for name, sensor in active_sensors.items()
            if isinstance(sensor, TerrainContact)
        }
        self._cameras = {
            name: sensor for name, sensor in active_sensors.items()
            if isinstance(sensor, FrameCamera)
        }
        self._camera_bvh_refit = any(camera.requires_bvh_refit for camera in self._cameras.values())
        self._solver = config.solver.build(
            self.scene,
            contacts,
            needs_torque,
            bool(self._contact_sensors),
            deterministic=config.deterministic,
            use_mujoco_contacts=config.collision == "mujoco",
        )
        self.contacts = self._solver.contacts
        self.sim = self._solver.solver
        self.observables = self._solver.observables
        self.state = self.model.state()
        self.state_next = self.model.state()
        self._physics_graphs = {}
        self._physics_warmed_up = False
        self.control = self.model.control()
        self.action_sizes = [
            entity.q.shape[1] for entity in self.scene.entities.values()
        ]
        self.action_dim = sum(self.action_sizes)
        self.control_dt = self.scene.control_dt
        self.profiler: StageProfiler | None = None

    def observation_symmetry_transform(
        self, groups: tuple[str, ...]
    ) -> SymmetryTransform:
        """Build a CPU transform in the order used to flatten those groups."""
        transforms = []
        for name in groups:
            try:
                transform = self.mdp.observations[name].symmetry_transform(self.scene)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot reflect observation group {name!r}: {exc}"
                ) from exc
            transforms.append(transform)
        return SymmetryTransform.cat(transforms)

    def _stream(self) -> wp.ScopedStream:
        return wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
        )

    def _profile_stage(self, name: str) -> AbstractContextManager[None]:
        return nullcontext() if self.profiler is None else self.profiler.stage(name)

    def _sample_cameras(
        self,
        *,
        reset: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> None:
        if self._camera_bvh_refit:
            self.model.bvh_refit_shapes(self.state)
            self.model.bvh_refit_particles(self.state)
        for name, camera in self._cameras.items():
            with self._profile_stage(f"camera/{name}"):
                camera.sample(self.state, reset=reset, reset_mask=reset_mask)

    def _simulate(
        self,
        state: newton.State,
        state_next: newton.State,
        *,
        profile: bool = False,
    ) -> tuple[newton.State, newton.State]:
        for sensor in self._contact_sensors.values():
            sensor.begin_step()
        for substep in range(self.config.substeps):
            final = substep == self.config.substeps - 1
            sample_contacts = any(
                sensor.track_history or final
                for sensor in self._contact_sensors.values()
            )
            with self._profile_stage("collision") if profile else nullcontext():
                state.clear_forces()
                if self.collision is not None:
                    self.collision.collide(state, self.contacts)
            with self._profile_stage("solver") if profile else nullcontext():
                self._solver.step(
                    state,
                    state_next,
                    self.control,
                    self.contacts,
                    self.config.sim_dt,
                    sample_contacts=sample_contacts,
                )
            if sample_contacts:
                with (
                    self._profile_stage("contact_sensor") if profile else nullcontext()
                ):
                    for sensor in self._contact_sensors.values():
                        if sensor.track_history or final:
                            sensor.update(
                                self.contacts,
                                self._solver.contact_observables,
                                state,
                                self.config.sim_dt,
                            )
            state, state_next = state_next, state
        return state, state_next

    def _advance(self) -> None:
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

    def _transition(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        result: TensorDict,
    ) -> TensorDict:
        """Expose independent successor snapshots for the configured output fields."""
        missing = [name for name in self.output_mdp_fields if name not in mdp_state]
        if missing:
            raise ValueError(f"Unknown output MDP fields: {missing}")
        result["mdp_state"] = mdp_state.select(*self.output_mdp_fields).clone()
        if self.output_physical_plan:
            result["physical_state"] = physical_state.select(
                *(
                    (entity, field)
                    for entity, fields in self.output_physical_plan.items()
                    for field in fields
                )
            ).clone()
        return result

    def _reset(
        self,
        env_ids: torch.Tensor | None,
        mdp_state: TensorDict | None,
    ) -> tuple[TensorDict, TensorDict]:
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
        self._solver.reset(self.state, ids, mask)
        joint_world = wp.to_torch(self.model.joint_world).long()
        dof_starts = wp.to_torch(self.model.joint_qd_start).long()
        dof_world = torch.repeat_interleave(joint_world, dof_starts[1:] - dof_starts[:-1])
        reset_dofs = mask[dof_world]
        if self.observables is not None:
            wp.to_torch(self.observables.qfrc_actuator)[reset_dofs] = 0
        for sensor in self._contact_sensors.values():
            sensor.reset(ids)
        # Reset velocities default to zero, independently of model defaults.
        wp.to_torch(self.state.joint_qd)[reset_dofs] = 0
        for name, pose in poses.items():
            entity = self.scene.entities[name]
            entity.write_joint_state(
                self.state, ids, joint_pos=pose.get("joint_pos"), joint_vel=pose.get("joint_vel")
            )
            entity.write_root_state(
                self.state, ids, root_pose=pose.get("root_pose"), root_twist=pose.get("root_twist")
            )
        joint_mask = mask[joint_world]
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
            if entity.velocity_mask is not None:
                wp.to_torch(self.control.joint_target_qd)[entity.qd[ids]] = 0.0
        self._sample_cameras(
            reset=True, reset_mask=None if env_ids is None else mask[:-1]
        )
        physical_state = self.scene.read(self.state, self.observables)
        result = TensorDict(
            {"observation": self.mdp.compute_observation(physical_state, next_mdp_state)},
            batch_size=[self.num_envs],
            device=self.device,
        )
        return next_mdp_state, self._transition(physical_state, next_mdp_state, result)

    def _step(self, action: torch.Tensor, mdp_state: TensorDict) -> tuple[TensorDict, TensorDict]:
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
        self._sample_cameras()
        with self._profile_stage("state_read"):
            physical_state = self.scene.read(self.state, self.observables)
        with self._profile_stage("mdp"):
            next_mdp_state, result = self.mdp.step(physical_state, mdp_state, action)
            transition = self._transition(physical_state, next_mdp_state, result)
        return next_mdp_state, transition


class StatefulEnvironment(_EnvironmentCore):
    """Own live physics buffers and MDP state. Resets are always explicit.

    Reset and step outputs include a snapshot of ``mdp_state.episode_step``
    for the returned observation; other MDP and physical fields are opt-in.
    """

    def __init__(self, config: EnvironmentConfig, mdp: MDP | None  =  None) -> None:
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
    Newton's joint and body states and persistent solver integration buffers.
    Steel uses point friction and recomputes its mass matrix each step, so its
    collision and constraint work does not require additional checkpoint history. Custom MDP
    terms must derive their results from the supplied physical snapshot and MDP state.
    """

    def __init__(self, config: EnvironmentConfig, mdp: MDP | None  =  None) -> None:
        super().__init__(config, mdp)
        if self.model.particle_count:
            raise ValueError("Simulator checkpoints currently support rigid scenes")

    def _checkpoint_views(self) -> dict[str, torch.Tensor]:
        arrays = {
            name: getattr(self.state, name)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd")
        }
        if self.observables is not None:
            arrays["qfrc_actuator"] = self.observables.qfrc_actuator
        views = {
            name: wp.to_torch(array).reshape(self.num_envs, -1)
            for name, array in arrays.items()
        }
        views.update(self._solver.checkpoint_views())
        views.update(
            {
                f"contact_{field}:{name}": view
                for name, sensor in self._contact_sensors.items()
                for field, view in sensor.checkpoint_views().items()
            }
        )
        views.update(
            {
                f"camera_{field}:{name}": view
                for name, sensor in self._cameras.items()
                for field, view in sensor.checkpoint_views().items()
            }
        )
        return views

    def _snapshot(self) -> TensorDict:
        return TensorDict(
            {name: view.clone() for name, view in self._checkpoint_views().items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _restore(self, sim_state: TensorDict) -> None:
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
