from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import newton
import warp as wp


class ObjectConfig(Protocol):
    def build(self) -> newton.ModelBuilder: ...


@dataclass(frozen=True)
class ObjectVariants:
    variants: tuple[ObjectConfig, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError("ObjectVariants requires at least one variant")
        if self.weights is not None and (
            len(self.weights) != len(self.variants)
            or any(w < 0 or not isfinite(w) for w in self.weights)
            or not 0 < sum(self.weights) < float("inf")
        ):
            raise ValueError("Supply one finite nonnegative weight per variant with positive total")


@dataclass(frozen=True, kw_only=True)
class _Procedural:
    position: tuple[float, float, float] = (0, 0, 0)
    orientation: tuple[float, float, float, float] = (0, 0, 0, 1)
    fixed: bool = False
    density: float = 500.0
    friction: float = 0.6

    def _root(self) -> tuple[newton.ModelBuilder, int, int]:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = self.density
        builder.default_shape_cfg.mu = self.friction
        pose = wp.transform(self.position, self.orientation)
        body = builder.add_link(xform=pose, label="base")
        if self.fixed:
            joint = builder.add_joint_fixed(-1, body, parent_xform=pose, label="root")
        else:
            joint = builder.add_joint_free(body, label="root")
        return builder, body, joint
