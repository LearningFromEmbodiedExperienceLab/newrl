"""Position, velocity, integrated position targets, and reference models."""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot, isfinite
from typing import TYPE_CHECKING, override

import newton
import torch
import warp as wp

from newrl.utils.string import resolve_scene_names

from .base import (
    Action, ActionChannel, ActionConfig, JointOutput, RobotTemplates,
    SymmetryChannel, Target, WorldIds,
)

if TYPE_CHECKING:
    from newrl.scene import Scene

JointValues = float | dict[str, float]


@dataclass(frozen=True)
class JointActionChannel(ActionChannel):
    entity: str
    joint: str
    target: str
    scale: float


@dataclass(frozen=True, kw_only=True)
class _JointConfig(ActionConfig):
    entity: str = "robot"
    joint_names: str | tuple[str, ...] = ".*"
    scale: JointValues = 1.0

    def __post_init__(self) -> None:
        if any(not isfinite(value) for value in _values(self.scale)):
            raise ValueError("Joint action scales must be finite")

    def resolve_joints(
        self, robots: RobotTemplates,
    ) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...]]:
        if self.entity not in robots:
            raise ValueError(f"Action requires robot {self.entity!r}")
        builder, joints = robots[self.entity]
        labels = [builder.joint_label[joint] for joint in joints]
        selected = tuple(resolve_scene_names(self.joint_names, labels))
        selected_labels = [labels[index] for index in selected]
        scales = _resolve_values(self.scale, selected_labels, "action scales")
        return selected, tuple(selected_labels), scales

    def channels(self, labels, scales, target) -> tuple[JointActionChannel, ...]:
        return tuple(
            JointActionChannel(f"{self.entity}/{label}:{target}", self.entity,
                               label.rsplit("/", 1)[-1], target, scale)
            for label, scale in zip(labels, scales, strict=True)
        )


@dataclass(frozen=True, kw_only=True)
class _FilteredJointConfig(_JointConfig):
    alpha: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.alpha <= 1:
            raise ValueError("Joint action alpha must be in (0, 1]")


@dataclass(frozen=True, kw_only=True)
class JointPosition(_FilteredJointConfig):
    """Command a scaled offset from the nominal joint position."""

    @override
    def build(self, robots: RobotTemplates, sim_dt: float) -> Action:
        joints, labels, scales = self.resolve_joints(robots)
        return _JointPosition(
            self.entity, joints, self.channels(labels, scales, "position"), self.alpha,
        )


@dataclass(frozen=True, kw_only=True)
class JointVelocity(_FilteredJointConfig):
    """Command an absolute joint velocity in joint units per second."""

    @override
    def build(self, robots: RobotTemplates, sim_dt: float) -> Action:
        joints, labels, scales = self.resolve_joints(robots)
        return _JointVelocity(
            self.entity, joints, self.channels(labels, scales, "velocity"), self.alpha,
        )


@dataclass(frozen=True, kw_only=True)
class JointPositionDelta(_FilteredJointConfig):
    """Integrate a scaled rate into a position target at each physics substep.

    Alpha filters the commanded rate. Reset starts the target at the reset
    joint position with zero rate.
    """

    @override
    def build(self, robots: RobotTemplates, sim_dt: float) -> Action:
        joints, labels, scales = self.resolve_joints(robots)
        return _JointPositionDelta(
            self.entity, joints, self.channels(labels, scales, "position_delta"), self.alpha,
        )


@dataclass(frozen=True, kw_only=True)
class JointReferenceModel(_JointConfig):
    """Advance a second-order reference and publish position and velocity each substep."""

    scale: JointValues = 0.5
    omega: JointValues = 20.0
    zeta: JointValues = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if any(not isfinite(value) or value <= 0 for value in _values(self.omega)):
            raise ValueError("Reference model omega must be finite and positive")
        if any(not isfinite(value) or value < 0 for value in _values(self.zeta)):
            raise ValueError("Reference model zeta must be finite and nonnegative")

    @override
    def build(self, robots: RobotTemplates, sim_dt: float) -> Action:
        joints, labels, scales = self.resolve_joints(robots)
        omega = _resolve_values(self.omega, labels, "omega")
        zeta = _resolve_values(self.zeta, labels, "zeta")
        # Stability bound for semi-implicit Euler.
        max_dt = min(2.0 / ((hypot(damping, 1.0) + damping) * frequency)
                     for frequency, damping in zip(omega, zeta, strict=True))
        if not 0 < sim_dt < max_dt:
            raise ValueError(f"Joint reference model requires sim_dt below {max_dt:g} s")
        return _ReferenceJointAction(
            self.entity, joints, self.channels(labels, scales, "reference"), omega, zeta,
        )


def _values(spec: JointValues):
    return spec.values() if isinstance(spec, dict) else (spec,)


def _resolve_values(spec: JointValues, labels, field: str) -> tuple[float, ...]:
    if not isinstance(spec, dict):
        return (float(spec),) * len(labels)
    indices = {}
    for pattern, value in spec.items():
        for index in resolve_scene_names(pattern, labels):
            if index in indices:
                raise ValueError(f"Overlapping {field} in joint action")
            indices[index] = float(value)
    if len(indices) != len(labels):
        raise ValueError(f"{field} must cover all selected joints")
    return tuple(indices[i] for i in range(len(labels)))


class _JointAction(Action):
    def __init__(
        self, entity: str, joints: tuple[int, ...], channels: tuple[JointActionChannel, ...],
    ) -> None:
        self.entity, self.joints, self.channels = entity, joints, channels
        self.target = channels[0].target
        self._symmetry_parameters = ((),) * len(joints)

    @override
    def _initialize(self, scene: "Scene", outputs: Mapping[Target, torch.Tensor]) -> None:
        self.robot = scene.robots[self.entity]
        self.outputs = outputs
        self.nominal = self.robot.default_position[:, self.joints]
        self.scale = self.nominal.new_tensor([channel.scale for channel in self.channels])

    def _reset_position(self, state: newton.State, ids: WorldIds) -> torch.Tensor:
        return wp.to_torch(state.joint_q)[self.robot.joint_q_indices[ids][:, self.joints]]

    def _symmetry_offsets(self) -> list[float]:
        return [0.0] * len(self.joints)

    @override
    def symmetry_channels(self) -> tuple[SymmetryChannel, ...]:
        channels = []
        for channel, offset, params in zip(
            self.channels, self._symmetry_offsets(), self._symmetry_parameters, strict=True
        ):
            mirror = self.robot.joint_mirror.get(channel.joint)
            if mirror is None:
                raise ValueError(f"No mirror action channel for {self.entity}/{channel.joint}")
            partner, sign = mirror
            # Compare the nominal offsets in a common orientation for each pair.
            if channel.joint == partner and sign == -1 and abs(offset) > 1.0e-8:
                raise ValueError("A self-mirrored signed position action requires zero nominal offset")
            if channel.joint > partner:
                offset *= sign
            channels.append(SymmetryChannel(
                (self.entity, channel.joint, self.target), (self.entity, partner, self.target),
                sign, (channel.scale, round(offset, 7), params),
            ))
        return tuple(channels)


@wp.kernel(enable_backward=False)
def _filter_target(
    desired: wp.array2d[float], filtered: wp.array2d[float], output: wp.array2d[float], alpha: float,
):
    world, joint = wp.tid()
    value = filtered[world, joint]
    value += alpha * (desired[world, joint] - value)
    filtered[world, joint] = value
    output[world, joint] = value


class _FilteredJointTarget(_JointAction):
    target_field: Target

    @override
    def __init__(
        self, entity: str, joints: tuple[int, ...], channels: tuple[JointActionChannel, ...],
        alpha: float,
    ) -> None:
        super().__init__(entity, joints, channels)
        self.alpha = alpha
        self.joint_outputs = (JointOutput(entity, joints, self.target_field),)
        self.every_substep = alpha < 1
        if self.every_substep:
            self._symmetry_parameters = ((alpha,),) * len(joints)

    @override
    def _initialize(self, scene: "Scene", outputs: Mapping[Target, torch.Tensor]) -> None:
        super()._initialize(scene, outputs)
        self.output = outputs[self.target_field]
        if not self.every_substep:
            self.command = self.output
            return
        self.command = torch.empty_like(self.nominal)
        self.filtered = torch.zeros_like(self.nominal)
        self.kernel_inputs = tuple(wp.from_torch(tensor) for tensor in (
            self.command, self.filtered, self.output,
        ))

    @abstractmethod
    def _filter_reset_value(self, state: newton.State, ids: WorldIds) -> torch.Tensor | float:
        """Initial value of the filtered target in the selected worlds."""

    @override
    def apply_substep(self, state: newton.State) -> None:
        wp.launch(_filter_target, tuple(self.filtered.shape),
                  inputs=[*self.kernel_inputs, self.alpha], device=state.joint_q.device)

    @override
    def reset(self, state: newton.State, ids: WorldIds) -> None:
        if not self.every_substep:
            return
        self.filtered[ids] = self._filter_reset_value(state, ids)
        self.output[ids] = self.filtered[ids]

    @override
    def checkpoint_views(self) -> Mapping[str, torch.Tensor]:
        return {"filtered_target": self.filtered} if self.every_substep else {}

    @override
    def restore_outputs(self) -> None:
        self.output.copy_(self.filtered)


class _JointPosition(_FilteredJointTarget):
    target_field = "position"

    @override
    def process_action(self, action: torch.Tensor) -> None:
        torch.addcmul(self.nominal, self.scale, action, out=self.command)

    @override
    def _filter_reset_value(self, state: newton.State, ids: WorldIds) -> torch.Tensor:
        return self._reset_position(state, ids)

    @override
    def _symmetry_offsets(self) -> list[float]:
        return self.nominal[0].cpu().tolist()

    @override
    def position_mapping(self) -> tuple[str, tuple[int, ...], torch.Tensor] | None:
        if not self.every_substep:
            return self.entity, self.joints, self.scale
        return None


class _JointVelocity(_FilteredJointTarget):
    target_field = "velocity"

    @override
    def process_action(self, action: torch.Tensor) -> None:
        torch.mul(self.scale, action, out=self.command)

    @override
    def _filter_reset_value(self, state: newton.State, ids: WorldIds) -> float:
        return 0.0


@wp.kernel(enable_backward=False)
def _integrate_position_delta(
    command: wp.array2d[float], rate: wp.array2d[float], position: wp.array2d[float],
    alpha: float, dt: float,
):
    world, joint = wp.tid()
    v = rate[world, joint]
    v += alpha * (command[world, joint] - v)
    rate[world, joint] = v
    position[world, joint] += dt * v


class _JointPositionDelta(_JointAction):
    every_substep = True

    @override
    def __init__(
        self, entity: str, joints: tuple[int, ...], channels: tuple[JointActionChannel, ...],
        alpha: float,
    ) -> None:
        super().__init__(entity, joints, channels)
        self.alpha = alpha
        self.joint_outputs = (JointOutput(entity, joints, "position"),)
        self._symmetry_parameters = ((alpha,),) * len(joints)

    @override
    def _initialize(self, scene: "Scene", outputs: Mapping[Target, torch.Tensor]) -> None:
        super()._initialize(scene, outputs)
        self.sim_dt = scene.sim_dt
        self.position = outputs["position"]
        self.command = torch.empty_like(self.nominal)
        self.rate = torch.zeros_like(self.nominal)
        self.kernel_inputs = tuple(wp.from_torch(tensor) for tensor in (
            self.command, self.rate, self.position,
        ))

    @override
    def process_action(self, action: torch.Tensor) -> None:
        torch.mul(self.scale, action, out=self.command)

    @override
    def apply_substep(self, state: newton.State) -> None:
        wp.launch(
            _integrate_position_delta, tuple(self.position.shape),
            inputs=[*self.kernel_inputs, self.alpha, self.sim_dt], device=state.joint_q.device,
        )

    @override
    def reset(self, state: newton.State, ids: WorldIds) -> None:
        self.position[ids] = self._reset_position(state, ids)
        self.rate[ids] = 0

    @override
    def checkpoint_views(self) -> Mapping[str, torch.Tensor]:
        return {"position_target": self.position, "filtered_rate": self.rate}


@wp.kernel(enable_backward=False)
def _advance_reference(
    desired: wp.array2d[float], position: wp.array2d[float], velocity: wp.array2d[float],
    omega_squared: wp.array[float], damping_rate: wp.array[float],
    target_q: wp.array2d[float], target_qd: wp.array2d[float], dt: float,
):
    world, joint = wp.tid()
    q = position[world, joint]
    v = velocity[world, joint]
    v += dt * (omega_squared[joint] * (desired[world, joint] - q) - damping_rate[joint] * v)
    q += dt * v
    position[world, joint] = q
    velocity[world, joint] = v
    target_q[world, joint] = q
    target_qd[world, joint] = v


class _ReferenceJointAction(_JointAction):
    every_substep = True

    @override
    def __init__(
        self, entity: str, joints: tuple[int, ...], channels: tuple[JointActionChannel, ...],
        omega: tuple[float, ...], zeta: tuple[float, ...],
    ) -> None:
        super().__init__(entity, joints, channels)
        self.joint_outputs = (
            JointOutput(entity, joints, "position"),
            JointOutput(entity, joints, "velocity"),
        )
        self._symmetry_parameters = tuple(zip(omega, zeta, strict=True))

    @override
    def _initialize(self, scene: "Scene", outputs: Mapping[Target, torch.Tensor]) -> None:
        super()._initialize(scene, outputs)
        self.sim_dt = scene.sim_dt
        self.position = torch.empty_like(self.nominal)
        self.velocity = torch.zeros_like(self.nominal)
        self.command = torch.empty_like(self.nominal)
        omega_squared = self.nominal.new_tensor([w * w for w, _ in self._symmetry_parameters])
        damping_rate = self.nominal.new_tensor([2 * d * w for w, d in self._symmetry_parameters])
        self.kernel_inputs = tuple(wp.from_torch(tensor) for tensor in (
            self.command, self.position, self.velocity, omega_squared, damping_rate,
            self.outputs["position"], self.outputs["velocity"],
        ))

    @override
    def process_action(self, action: torch.Tensor) -> None:
        torch.addcmul(self.nominal, self.scale, action, out=self.command)

    @override
    def apply_substep(self, state: newton.State) -> None:
        wp.launch(_advance_reference, tuple(self.position.shape),
                  inputs=[*self.kernel_inputs, self.sim_dt], device=state.joint_q.device)

    @override
    def reset(self, state: newton.State, ids: WorldIds) -> None:
        self.position[ids] = self._reset_position(state, ids)
        self.velocity[ids] = 0
        self.outputs["position"][ids] = self.position[ids]
        self.outputs["velocity"][ids] = self.velocity[ids]

    @override
    def checkpoint_views(self) -> Mapping[str, torch.Tensor]:
        return {"reference_position": self.position, "reference_velocity": self.velocity}

    @override
    def restore_outputs(self) -> None:
        self.outputs["position"].copy_(self.position)
        self.outputs["velocity"].copy_(self.velocity)

    @override
    def _symmetry_offsets(self) -> list[float]:
        return self.nominal[0].cpu().tolist()
