"""Per-world physical parameters sampled when a simulator is built."""

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp

if TYPE_CHECKING:
    from .scene import RobotEntity, Scene


@dataclass(frozen=True, kw_only=True)
class _Parameter:
    bounds: tuple[float, float]
    entity: str = "robot"

    def __post_init__(self) -> None:
        low, high = self.bounds
        if not (0 < low <= high and isfinite(high)):
            raise ValueError("Randomization bounds must be positive, finite, and ordered")


@dataclass(frozen=True, kw_only=True)
class BodyMassScale(_Parameter):
    """Scale the mass and inertia of each matched body together."""

    body_names: str | tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ShapeFriction(_Parameter):
    """Set sliding friction on collision shapes attached to matched bodies."""

    body_names: str | tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class JointGainScale(_Parameter):
    """Scale both Newton joint target gains for each matched joint."""

    joint_names: str | tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class DomainRandomizationConfig:
    mass: BodyMassScale | None = None
    friction: ShapeFriction | None = None
    gains: JointGainScale | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**64:
            raise ValueError("Randomization seed must be in [0, 2**64)")
        if self.mass is None and self.friction is None and self.gains is None:
            raise ValueError("Configure at least one domain randomization parameter")


@dataclass
class _ModelWrite:
    target: torch.Tensor
    ids: torch.Tensor
    baseline: torch.Tensor | None = None
    inverse: bool = False

    def apply(self, value: torch.Tensor) -> None:
        if self.baseline is not None:
            if self.inverse:
                value = value.reciprocal()
            value = value.reshape(*value.shape, *((1,) * (self.baseline.ndim - value.ndim)))
            value = self.baseline * value
        self.target[self.ids] = value


class DomainRandomization:
    """Bind scene indices once and edit Newton's existing model arrays in place."""

    def __init__(self, scene: "Scene", config: DomainRandomizationConfig) -> None:
        model = scene.model
        device = torch.device(wp.device_to_torch(model.device))
        generator = torch.Generator().manual_seed(config.seed)
        self.values: dict[str, torch.Tensor] = {}
        self._writes: dict[str, tuple[_ModelWrite, ...]] = {}
        self._bounds: dict[str, tuple[float, float]] = {}
        self.flags: int = 0

        if (term := config.mass) is not None:
            entity = scene.entities[term.entity]
            indices, _ = entity.find_bodies(term.body_names)
            ids = entity.body_indices[:, indices]
            writes = []
            for name in ("body_mass", "body_inertia", "body_inv_mass", "body_inv_inertia"):
                target = wp.to_torch(getattr(model, name))
                writes.append(_ModelWrite(target, ids, target[ids], name.startswith("body_inv_")))
            self._add("body_mass_scale", term, entity, tuple(writes), generator)
            self.flags |= newton.ModelFlags.BODY_INERTIAL_PROPERTIES

        if (term := config.friction) is not None:
            entity = scene.entities[term.entity]
            indices, _ = entity.find_bodies(term.body_names)
            bodies = entity.body_indices[:, indices]
            owners = wp.to_torch(model.shape_body).long()
            flags = wp.to_torch(model.shape_flags)
            selected_bodies = torch.zeros(model.body_count, dtype=torch.bool, device=device)
            selected_bodies[bodies] = True
            selected = (
                (owners >= 0)
                & selected_bodies[owners.clamp_min(0)]
                & ((flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)
            )
            ids = selected.nonzero().reshape(model.world_count, -1)
            if ids.shape[1] == 0:
                raise ValueError("Shape friction selector matched no collision shapes")
            target = wp.to_torch(model.shape_material_mu)
            self._add("shape_friction", term, entity, (_ModelWrite(target, ids),), generator)
            self.flags |= newton.ModelFlags.SHAPE_PROPERTIES

        if (term := config.gains) is not None:
            entity = scene.entities[term.entity]
            indices, _ = entity.find_joints(term.joint_names)
            ids = entity.joint_qd_indices[:, indices]
            writes = []
            for name in ("joint_target_ke", "joint_target_kd"):
                target = wp.to_torch(getattr(model, name))
                writes.append(_ModelWrite(target, ids, target[ids]))
            self._add("joint_gain_scale", term, entity, tuple(writes), generator)
            self.flags |= newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES

        self._apply()

    def _add(
        self, name: str, term: _Parameter, entity: "RobotEntity",
        writes: tuple[_ModelWrite, ...], generator: torch.Generator,
    ) -> None:
        low, high = term.bounds
        self.values[name] = torch.empty(writes[0].ids.shape).uniform_(
            low, high, generator=generator
        ).to(writes[0].target.device)
        entity.model_parameters[name] = self.values[name]
        self._writes[name] = writes
        self._bounds[name] = term.bounds

    def _apply(self) -> None:
        for name, writes in self._writes.items():
            for write in writes:
                write.apply(self.values[name])

    def restore(self, values: dict[str, torch.Tensor]) -> bool:
        """Restore samples after checkpoint shape validation. Report model changes."""
        if all(torch.equal(value, self.values[name]) for name, value in values.items()):
            return False
        for name, value in values.items():
            low, high = self._bounds[name]
            if not ((value >= low) & (value <= high)).all():
                raise ValueError(f"Simulator checkpoint randomization field {name!r} is out of range")
        for name, value in values.items():
            self.values[name].copy_(value)
        self._apply()
        return True
