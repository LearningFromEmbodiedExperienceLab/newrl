from dataclasses import dataclass

import newton

from .config import _Procedural


@dataclass(frozen=True)
class BoxConfig(_Procedural):
    half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1)

    def build(self) -> newton.ModelBuilder:
        b, body, root = self._root()
        x, y, z = self.half_extents
        b.add_shape_box(body, hx=x, hy=y, hz=z)
        b.add_articulation([root])
        return b


@dataclass(frozen=True)
class SphereConfig(_Procedural):
    radius: float = 0.1

    def build(self) -> newton.ModelBuilder:
        b, body, root = self._root()
        b.add_shape_sphere(body, radius=self.radius)
        b.add_articulation([root])
        return b


@dataclass(frozen=True)
class CylinderConfig(_Procedural):
    radius: float = 0.1
    half_height: float = 0.15

    def build(self) -> newton.ModelBuilder:
        b, body, root = self._root()
        b.add_shape_cylinder(body, radius=self.radius, half_height=self.half_height)
        b.add_articulation([root])
        return b
