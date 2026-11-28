"""Action interfaces, simulator binding, and policy action layout."""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import newton
import torch
import warp as wp

from newrl.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    from newrl.scene import Scene

Target = Literal["position", "velocity"]
WorldIds = torch.Tensor | slice
RobotTemplates = Mapping[str, tuple[newton.ModelBuilder, list[int]]]


@dataclass(frozen=True)
class ActionChannel:
    name: str


@dataclass(frozen=True)
class JointOutput:
    entity: str
    joints: tuple[int, ...]
    target: Target


@dataclass(frozen=True)
class SymmetryChannel:
    key: Hashable
    partner: Hashable
    sign: int
    signature: Hashable = ()


class Action(ABC):
    """One policy action component, built before the Newton model is finalized.

    Declare channels and joint outputs at construction. _initialize receives
    the scene and writable views of the declared outputs after replication.
    The layout publishes those views to Newton.
    """

    channels: tuple[ActionChannel, ...]
    joint_outputs: tuple[JointOutput, ...] = ()
    every_substep: bool = False

    def _initialize(self, scene: "Scene", outputs: Mapping[Target, torch.Tensor]) -> None:
        """Bind output views keyed by target field and allocate any runtime state."""

    @abstractmethod
    def process_action(self, action: torch.Tensor) -> None:
        """Consume this component's slice of the policy action once per control step."""

    def apply_substep(self, state: newton.State) -> None:
        """Update outputs before each solver substep if every_substep is set.

        Read scene.sim_dt during initialization if needed. Implementations
        must be CUDA graph safe.
        """

    def reset(self, state: newton.State, ids: WorldIds) -> None:
        """Reset selected runtime rows and refresh their stateful outputs."""

    def checkpoint_views(self) -> Mapping[str, torch.Tensor]:
        """Expose persistent state tensors with stable storage for replay."""
        return {}

    def restore_outputs(self) -> None:
        """Refresh output views after the layout restores this action's checkpoint tensors."""

    def symmetry_channels(self) -> tuple[SymmetryChannel, ...]:
        raise ValueError("Action component does not declare symmetry")

    def position_mapping(self) -> tuple[str, tuple[int, ...], torch.Tensor] | None:
        """Describe a direct invertible position mapping, when available."""
        return None


class ActionConfig(ABC):
    """Reusable configuration for one action component."""

    @abstractmethod
    def build(self, robots: RobotTemplates, sim_dt: float) -> Action:
        """Create a fresh action with channels and actuator targets resolved."""


@wp.kernel(enable_backward=False)
def _write_targets(
    indices: wp.array2d[wp.int64], values: wp.array2d[float], targets: wp.array[float],
):
    world, channel = wp.tid()
    targets[indices[world, channel]] = values[world, channel]


@dataclass(frozen=True)
class _ControlWrite:
    field: str
    values: torch.Tensor
    indices_wp: wp.array
    values_wp: wp.array

    def apply(self, control: newton.Control) -> None:
        target = getattr(control, self.field)
        wp.launch(
            _write_targets, tuple(self.values.shape),
            inputs=[self.indices_wp, self.values_wp, target], device=target.device,
        )


class ActionLayout:
    """Dispatch action slices and aggregate joint outputs into Newton control buffers."""

    def __init__(
        self, actions: Mapping[str, ActionConfig], robots: RobotTemplates, sim_dt: float,
    ) -> None:
        """Resolve action channels and set actuator modes on the robot templates."""
        components: dict[str, Action] = {}
        ownership: dict[tuple[str, int], set[Target]] = {}
        for name, config in actions.items():
            if not name or "/" in name:
                raise ValueError("Action component names must be nonempty and omit slashes")
            component = config.build(robots, sim_dt)
            output_fields = set()
            for output in component.joint_outputs:
                if output.target in output_fields:
                    raise ValueError(f"Action {name!r} declares multiple batches for {output.target}")
                output_fields.add(output.target)
                builder, joints = robots[output.entity]
                for index in output.joints:
                    joint = joints[index]
                    fields = ownership.setdefault((output.entity, joint), set())
                    if output.target in fields:
                        label = f"{output.entity}/{builder.joint_label[joint]}"
                        raise ValueError(f"Duplicate {output.target} action target for {label}")
                    fields.add(output.target)
            components[name] = component

        for (entity, joint), fields in ownership.items():
            builder, _ = robots[entity]
            dof = builder.joint_qd_start[joint]
            ke, kd = builder.joint_target_ke[dof], builder.joint_target_kd[dof]
            label = f"{entity}/{builder.joint_label[joint]}"
            if ("position" in fields and ke <= 0) or ("velocity" in fields and kd <= 0):
                raise ValueError(f"Action target has no matching actuator gain on {label}")
            if "velocity" in fields:
                if ke > 0 and "position" not in fields:
                    raise ValueError(f"Velocity-only action requires zero stiffness on {label}")
                mode = (
                    newton.JointTargetMode.POSITION_VELOCITY
                    if "position" in fields else newton.JointTargetMode.VELOCITY
                )
            else:
                mode = newton.JointTargetMode.POSITION
            builder.joint_target_mode[dof] = int(mode)

        self._components = components
        self.channels = tuple(
            channel for component in components.values() for channel in component.channels
        )
        self._widths = tuple(len(component.channels) for component in components.values())
        self.slices: dict[str, slice] = {}
        offset = 0
        for name, width in zip(components, self._widths, strict=True):
            self.slices[name] = slice(offset, offset + width)
            offset += width

    def _initialize(self, scene: "Scene") -> None:
        """Bind runtime buffers after the scene has built its replicated entities."""
        robots = self._robots = scene.robots
        components = self._components
        outputs = {name: {} for name in components}
        target_views = {name: [] for name in components}
        self._control_step_writes: list[_ControlWrite] = []
        self._substep_writes: list[_ControlWrite] = []
        for every_substep, writes in (
            (False, self._control_step_writes), (True, self._substep_writes),
        ):
            for target, field, index_field in (
                ("position", "joint_target_q", "joint_target_indices"),
                ("velocity", "joint_target_qd", "joint_qd_indices"),
            ):
                entries = [
                    (name, getattr(robots[output.entity], index_field)[:, output.joints])
                    for name, component in components.items() if component.every_substep == every_substep
                    for output in component.joint_outputs if output.target == target
                ]
                if not entries:
                    continue
                indices = torch.cat([tensor for _, tensor in entries], dim=-1).contiguous()
                values = torch.empty(indices.shape, dtype=torch.float32, device=indices.device)
                writes.append(
                    _ControlWrite(field, values, wp.from_torch(indices), wp.from_torch(values))
                )
                start = 0
                for name, indices_for_component in entries:
                    end = start + indices_for_component.shape[1]
                    outputs[name][target] = values[:, start:end]
                    target_views[name].append((field, indices_for_component, outputs[name][target]))
                    start = end
        for name, component in self._components.items():
            component._initialize(scene, outputs[name])
        self._substep_components = tuple(
            component for component in self._components.values() if component.every_substep
        )
        checkpoint_views = {
            name: component.checkpoint_views() for name, component in self._components.items()
        }
        self._stateful_components = tuple(
            self._components[name] for name, views in checkpoint_views.items() if views
        )
        self._stateful_targets = tuple(
            target for name, views in checkpoint_views.items() if views for target in target_views[name]
        )
        self._checkpoint_views = {
            f"action/{name}/{key}": view
            for name, views in checkpoint_views.items()
            for key, view in views.items()
        }

    @property
    def dim(self) -> int:
        return len(self.channels)

    def process_actions(self, action: torch.Tensor) -> None:
        if len(self._components) == 1:
            next(iter(self._components.values())).process_action(action)
            return
        for component, values in zip(
            self._components.values(), action.split(self._widths, dim=-1), strict=True
        ):
            component.process_action(values)

    def apply_substep(
        self, control: newton.Control, state: newton.State, *, first_substep: bool,
    ) -> None:
        """Publish held targets once and update dynamic targets before each solve."""
        if first_substep:
            for write in self._control_step_writes:
                write.apply(control)
        for component in self._substep_components:
            component.apply_substep(state)
        for write in self._substep_writes:
            write.apply(control)

    def checkpoint_views(self) -> Mapping[str, torch.Tensor]:
        return self._checkpoint_views

    def restore_outputs(self, control: newton.Control) -> None:
        for component in self._stateful_components:
            component.restore_outputs()
        self._publish_stateful_targets(control, slice(None))

    def _publish_stateful_targets(self, control: newton.Control, ids: WorldIds) -> None:
        for field, indices, values in self._stateful_targets:
            wp.to_torch(getattr(control, field))[indices[ids]] = values[ids]

    def reset(self, control: newton.Control, state: newton.State, ids: WorldIds) -> None:
        for robot in self._robots.values():
            robot.write_joint_target(
                control, ids, joint_pos=robot.default_position[ids], joint_vel=0.0,
            )
        for component in self._components.values():
            component.reset(state, ids)
        self._publish_stateful_targets(control, ids)

    def symmetry_transform(self) -> SymmetryTransform:
        channels = []
        for name, component in self._components.items():
            mapping = component.symmetry_channels()
            selection = self.slices[name]
            if len(mapping) != selection.stop - selection.start:
                raise ValueError(f"Wrong symmetry width for action {name!r}")
            channels.extend(mapping)
        lookup = {channel.key: i for i, channel in enumerate(channels)}
        if len(lookup) != len(channels):
            raise ValueError("Action symmetry keys must be unique")
        indices = []
        for channel in channels:
            if channel.partner not in lookup:
                raise ValueError(f"No mirror action channel for {channel.key!r}")
            index = lookup[channel.partner]
            if channel.signature != channels[index].signature:
                raise ValueError("Paired action channels require matching scales, offsets, and filters")
            indices.append(index)
        return SymmetryTransform(indices, [channel.sign for channel in channels])

    def inverse_position_mapping(self, entity: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return policy-ordered joint indices, nominal offsets and inverse scales."""
        order, scales = [], []
        robot = self._robots[entity]
        for component in self._components.values():
            mapping = component.position_mapping()
            if mapping is None or mapping[0] != entity:
                raise ValueError("Residual actions require direct position control")
            order.extend(mapping[1])
            scales.append(mapping[2])
        if not scales or sorted(order) != list(range(len(robot.joint_names))):
            raise ValueError("Residual actions require full position control")
        indices = torch.tensor(order, device=robot.default_position.device, dtype=torch.long)
        scale = torch.cat(scales)
        if torch.any(scale <= 0):
            raise ValueError("Residual actions require positive action scales")
        return indices, robot.default_position[0, indices].detach().clone(), scale.reciprocal()
