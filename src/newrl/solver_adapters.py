"""Adapters for solver-specific stepping, reset, and checkpoint behavior."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol, TYPE_CHECKING

import newton
import torch
import warp as wp

from .steel import reconstruct_augmented_drive_torque

if TYPE_CHECKING:
    from newrl.scene import Scene


class SolverAdapter(Protocol):
    solver: object
    contacts: newton.Contacts | None
    observables: newton.solvers.SolverMuJoCo.Observables | SimpleNamespace | None
    contact_observables: newton.solvers.SolverMuJoCo.Observables | newton.solvers.SolverSteel.Observables | None

    def step(
        self, state: newton.State, state_next: newton.State, control: newton.Control, contacts: newton.Contacts | None, dt: float, *, sample_contacts: bool
    ) -> None: ...

    def reset(self, state: newton.State, ids: torch.Tensor | slice, mask: torch.Tensor) -> None: ...

    def checkpoint_views(self) -> dict[str, torch.Tensor]: ...

    def refresh_model(self, flags: int) -> None: ...


class MujocoSim:
    def refresh_model(self, flags: int) -> None:
        self.solver.notify_model_changed(flags)

    def __init__(
        self,
        config: "MujocoSimConfig",
        scene: "Scene",
        contacts: newton.Contacts | None,
        needs_torque: bool,
        needs_contacts: bool,
        *,
        deterministic: bool,
        use_mujoco_contacts: bool=False,
    ) -> None:
        self.model = scene.model
        self.num_envs = scene.model.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            jacobian=config.jacobian,
            solver=config.algorithm,
            integrator=config.integrator,
            cone=config.cone,
            impratio=config.impratio,
            iterations=config.iterations,
            ls_iterations=config.ls_iterations,
            nconmax=config.nconmax,
            njmax=config.njmax,
            use_mujoco_contacts=use_mujoco_contacts,
            enable_sleeping=False,
            deterministic=wp.DeterministicMode.RUN_TO_RUN if deterministic else None,
        )
        self.observables = None
        self.contact_observables = None
        if use_mujoco_contacts and needs_contacts:
            contacts = newton.Contacts(
                self.solver.get_max_contact_count(), 0, device=self.model.device
            )
        self.contacts = contacts
        if needs_contacts:
            kinds = {self.solver.ObservableKind.CONTACT_F}
            if needs_torque:
                kinds.add(self.solver.ObservableKind.QFRC_ACTUATOR)
            self.contact_observables = self.solver.observables(kinds, contacts=contacts)
            if needs_torque:
                self.observables = self.contact_observables.select(
                    {self.solver.ObservableKind.QFRC_ACTUATOR}
                )
        elif needs_torque:
            self.observables = self.solver.observables(
                {self.solver.ObservableKind.QFRC_ACTUATOR}
            )
        self._reset_defaults = {
            name: wp.to_torch(array).clone()
            for name in ("history", "mocap_pos", "mocap_quat")
            if (array := getattr(self.solver.mjw_data, name)).size
        }

    def step(
        self,
        state: newton.State,
        state_next: newton.State,
        control: newton.Control,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        sample_contacts: bool,
    ) -> None:
        observables = self.contact_observables if sample_contacts else self.observables
        self.solver.step(
            state, state_next, control, contacts, dt, observables=observables
        )

    def reset(self, state: newton.State, ids: torch.Tensor | slice, mask: torch.Tensor) -> None:
        self.solver.reset(state, world_mask=wp.from_torch(mask, dtype=wp.bool))
        wp.to_torch(self.solver.mjw_data.time)[ids] = 0
        for name, defaults in self._reset_defaults.items():
            wp.to_torch(getattr(self.solver.mjw_data, name))[ids] = defaults[ids]

    def checkpoint_views(self) -> dict[str, torch.Tensor]:
        return {
            name: wp.to_torch(array).reshape(self.num_envs, -1)
            for name in (
                "qacc_warmstart",
                "act",
                "time",
                "history",
                "mocap_pos",
                "mocap_quat",
            )
            if (array := getattr(self.solver.mjw_data, name)).size
        }


class SteelSim:
    def refresh_model(self, flags: int) -> None:
        self.solver.notify_model_changed(flags)

    def __init__(
        self, config: "SteelSimConfig", scene: "Scene", contacts: newton.Contacts | None, needs_torque: bool, needs_contacts: bool, *, deterministic: bool
    ) -> None:
        self.model = scene.model
        self.contacts = contacts
        self.num_envs = scene.model.world_count
        self.model.rigid_contact_max = contacts.rigid_contact_max
        self.solver = newton.solvers.SolverSteel(
            self.model,
            iterations=config.iterations,
            articulation_constraint_max=config.articulation_constraint_max,
            enable_joint_limits=True,
            drive_mode="augmented",
            friction_anchor_beta=0.0,
            contact_warmstart=False,
            update_mass_matrix_interval=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN if deterministic else None,
        )
        self.observables = (
            SimpleNamespace(
                qfrc_actuator=wp.zeros(
                    self.model.joint_dof_count,
                    dtype=wp.float32,
                    device=self.model.device,
                )
            )
            if needs_torque
            else None
        )
        self.contact_observables = (
            self.solver.observables(
                {self.solver.ObservableKind.CONTACT_F}, contacts=contacts
            )
            if needs_contacts
            else None
        )
        self._indices = tuple(
            wp.from_torch(
                torch.cat(
                    [
                        getattr(entity, name).reshape(-1)
                        for entity in scene.entities.values()
                    ]
                )
                .int()
                .contiguous(),
                dtype=wp.int32,
            )
            for name in ("joint_q_indices", "joint_qd_indices", "joint_target_indices")
        )

    def step(
        self,
        state: newton.State,
        state_next: newton.State,
        control: newton.Control,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        sample_contacts: bool,
    ) -> None:
        self.solver.step(
            state,
            state_next,
            control,
            contacts,
            dt,
            observables=self.contact_observables if sample_contacts else None,
        )
        if self.observables is not None:
            wp.launch(
                reconstruct_augmented_drive_torque,
                dim=self._indices[0].size,
                inputs=[
                    *self._indices,
                    state.joint_q,
                    state.joint_qd,
                    state_next.joint_qd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_effort_limit,
                    dt,
                    self.observables.qfrc_actuator,
                ],
                device=self.model.device,
            )

    def reset(self, state: newton.State, ids: torch.Tensor | slice, mask: torch.Tensor) -> None:
        self.solver.reset(state, world_mask=wp.from_torch(mask, dtype=wp.bool))
        joint_world = wp.to_torch(self.model.joint_world).long()
        for name, starts in (
            ("joint_q", self.model.joint_q_start),
            ("joint_qd", self.model.joint_qd_start),
        ):
            offsets = wp.to_torch(starts).long()
            worlds = torch.repeat_interleave(joint_world, offsets[1:] - offsets[:-1])
            target = wp.to_torch(getattr(state, name))
            target[mask[worlds]] = wp.to_torch(getattr(self.model, name))[mask[worlds]]

    def checkpoint_views(self) -> dict[str, torch.Tensor]:
        return {
            "constraint_overflow": wp.to_torch(self.solver.constraint_overflow)[
                : self.num_envs
            ].reshape(self.num_envs, 1)
        }


@dataclass
class MujocoSimConfig:
    jacobian: str = "sparse"
    algorithm: str = "newton"
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 100.0
    iterations: int = 30
    ls_iterations: int = 40
    nconmax: int = 80
    njmax: int = 100

    def build(
        self,
        scene: "Scene",
        contacts: newton.Contacts | None,
        needs_torque: bool,
        needs_contacts: bool,
        *,
        deterministic: bool,
        use_mujoco_contacts: bool=False,
    ) -> SolverAdapter:
        return MujocoSim(
            self,
            scene,
            contacts,
            needs_torque,
            needs_contacts,
            deterministic=deterministic,
            use_mujoco_contacts=use_mujoco_contacts,
        )


@dataclass
class SteelSimConfig:
    iterations: int = 12
    articulation_constraint_max: int = 128

    def build(
        self,
        scene: "Scene",
        contacts: newton.Contacts | None,
        needs_torque: bool,
        needs_contacts: bool,
        *,
        deterministic: bool,
        use_mujoco_contacts: bool=False,
    ) -> SolverAdapter:
        if use_mujoco_contacts:
            raise ValueError("collision=mujoco requires the MuJoCo solver")
        return SteelSim(
            self,
            scene,
            contacts,
            needs_torque,
            needs_contacts,
            deterministic=deterministic,
        )


from .utils.config import cs

cs.store("mujoco", MujocoSimConfig, group="solver")
cs.store("steel", SteelSimConfig, group="solver")
