from dataclasses import dataclass
from math import pi

import newton
import warp as wp

from .config import _Procedural


def _pose(position: tuple[float, float, float]) -> wp.transform:
    return wp.transform(position, wp.quat_identity())


@dataclass(frozen=True)
class TableConfig(_Procedural):
    """Four-legged table with its origin on the floor and height at the top surface."""

    fixed: bool = True
    width: float = 0.9
    depth: float = 0.6
    height: float = 0.72

    def build(self) -> newton.ModelBuilder:
        builder, body, root = self._root()
        half_depth, half_width = self.depth / 2, self.width / 2
        thickness = 0.04
        leg_radius = 0.025
        leg_half_height = (self.height - thickness) / 2
        builder.add_shape_box(body, hx=half_depth, hy=half_width, hz=thickness / 2,
                              xform=_pose((0, 0, self.height - thickness / 2)))
        for x in (-1, 1):
            for y in (-1, 1):
                center = (x * (half_depth - leg_radius), y * (half_width - leg_radius), leg_half_height)
                builder.add_shape_box(body, hx=leg_radius, hy=leg_radius, hz=leg_half_height,
                                      xform=_pose(center))
        builder.add_articulation([root])
        return builder


@dataclass(frozen=True)
class ChairConfig(_Procedural):
    seat_height: float = 0.45
    width: float = 0.46
    depth: float = 0.44

    def build(self) -> newton.ModelBuilder:
        b, body, root = self._root()
        x, y, h = self.depth / 2, self.width / 2, self.seat_height
        b.add_shape_box(body, hx=x, hy=y, hz=0.02, xform=_pose((0, 0, h)))
        for dx in (-1, 1):
            for dy in (-1, 1):
                b.add_shape_box(body, hx=0.02, hy=0.02, hz=h / 2,
                                xform=_pose((dx * (x - 0.02), dy * (y - 0.02), h / 2)))
        b.add_shape_box(body, hx=0.02, hy=y, hz=0.2, xform=_pose((-x, 0, h + 0.2)))
        b.add_articulation([root])
        return b


@dataclass(frozen=True)
class DoorConfig(_Procedural):
    fixed: bool = True
    width: float = 0.8
    height: float = 1.9

    def build(self) -> newton.ModelBuilder:
        b, frame, root = self._root()
        h = self.height / 2
        b.add_shape_box(frame, hx=0.04, hy=0.04, hz=h, xform=_pose((-0.04, 0, h)))
        panel = b.add_link(label="panel")
        b.add_shape_box(panel, hx=self.width / 2, hy=0.02, hz=h,
                        xform=_pose((self.width / 2, 0, h)))
        b.add_shape_box(panel, hx=0.06, hy=0.035, hz=0.012,
                        xform=_pose((self.width - 0.1, -0.05, h)))
        hinge = b.add_joint_revolute(frame, panel, axis=newton.Axis.Z,
                                     limit_lower=0, limit_upper=pi / 2, label="hinge")
        b.add_articulation([root, hinge])
        return b


@dataclass(frozen=True)
class DrawerConfig(_Procedural):
    fixed: bool = True
    depth: float = 0.5
    width: float = 0.4

    def build(self) -> newton.ModelBuilder:
        b, frame, root = self._root()
        x, y = self.depth / 2, self.width / 2
        b.add_shape_box(frame, hx=x, hy=y, hz=0.02, xform=_pose((0, 0, 0.02)))
        tray = b.add_link(label="tray")
        b.add_shape_box(tray, hx=x, hy=y, hz=0.015, xform=_pose((0, 0, 0.06)))
        for dy in (-y, y):
            b.add_shape_box(tray, hx=x, hy=0.01, hz=0.05, xform=_pose((0, dy, 0.11)))
        for dx in (-x, x):
            b.add_shape_box(tray, hx=0.01, hy=y, hz=0.05, xform=_pose((dx, 0, 0.11)))
        slide = b.add_joint_prismatic(frame, tray, axis=newton.Axis.X, limit_lower=0,
                                      limit_upper=0.8 * self.depth, label="slide")
        b.add_articulation([root, slide])
        return b
