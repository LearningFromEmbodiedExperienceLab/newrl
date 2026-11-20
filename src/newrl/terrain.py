"""Static terrain geometry and optional spawn anchors for each replicated world."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import newton
import numpy as np
from newton._src.utils.mesh import load_meshes_from_file


@dataclass(frozen=True, kw_only=True)
class TerrainConfig:
    """Spawn locations are world-space ground anchors in meters.

    The MDP may sample an anchor and add it to the nominal root translation.
    Coordinates refer to the shared terrain seen by each Newton world.
    """

    spawn_locations: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self):
        for location in self.spawn_locations:
            if len(location) != 3 or not all(isfinite(value) for value in location):
                raise ValueError("Spawn locations must contain finite XYZ coordinates")

    def build(self, builder: newton.ModelBuilder) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class PlaneTerrainConfig(TerrainConfig):
    def build(self, builder: newton.ModelBuilder) -> None:
        builder.add_ground_plane()


@dataclass(frozen=True)
class MeshTerrainConfig(TerrainConfig):
    """Static mesh, file path, or factory called once when the scene is built."""

    source: newton.Mesh | Path | str | Callable[[], newton.Mesh]

    def build(self, builder: newton.ModelBuilder) -> None:
        if isinstance(self.source, (str, Path)):
            meshes = load_meshes_from_file(
                str(self.source), maxhullvert=64, compute_inertia=False
            )
        else:
            mesh = self.source() if callable(self.source) else self.source
            meshes = [mesh]
        if not meshes or any(
            not isinstance(mesh, newton.Mesh) or not len(mesh.indices)
            for mesh in meshes
        ):
            raise ValueError(
                "Mesh terrain requires at least one nonempty Newton triangle mesh"
            )
        for index, mesh in enumerate(meshes):
            label = "ground_mesh" if len(meshes) == 1 else f"ground_mesh_{index}"
            builder.add_shape_mesh(body=-1, mesh=mesh, label=label)


@dataclass(frozen=True)
class FlatMeshTerrainConfig(TerrainConfig):
    """The flat slab from the mesh collision benchmark."""

    resolution: int = 1
    size: float = 20.0
    thickness: float = 1.0

    def build(self, builder: newton.ModelBuilder) -> None:
        mesh = flat_ground_mesh(self.resolution, self.size, self.thickness)
        builder.add_shape_mesh(body=-1, mesh=mesh, label="ground_mesh")


def flat_ground_mesh(resolution: int, size: float, thickness: float) -> newton.Mesh:
    """A flat slab whose top is z=0 and whose bottom is ``thickness`` below.

    The top is a ``resolution`` by ``resolution`` grid, ``2 * resolution**2``
    triangles. ``resolution=1`` is two triangles. The bottom and each of the
    four sides are two triangles, using only the slab corners, whatever the
    top resolution is. Winding is counter-clockwise from the outside.
    """
    if resolution < 1:
        raise ValueError(f"mesh resolution must be >= 1, got {resolution}")
    if size <= 0.0:
        raise ValueError(f"mesh size must be positive, got {size}")
    if thickness <= 0.0:
        raise ValueError(f"mesh thickness must be positive, got {thickness}")
    n = resolution + 1
    half = 0.5 * size
    axis = np.linspace(-half, half, n, dtype=np.float32)
    xs, ys = np.meshgrid(axis, axis, indexing="xy")
    top = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n, dtype=np.float32)])
    # Corner order: (-x,-y), (+x,-y), (-x,+y), (+x,+y).
    bottom = np.array(
        [
            [-half, -half, -thickness],
            [half, -half, -thickness],
            [-half, half, -thickness],
            [half, half, -thickness],
        ],
        dtype=np.float32,
    )
    vertices = np.vstack([top, bottom])
    base = n * n
    b00, b10, b01, b11 = base, base + 1, base + 2, base + 3
    t00, t10, t01, t11 = 0, resolution, resolution * n, n * n - 1
    faces = []
    for j in range(resolution):
        for i in range(resolution):
            a = j * n + i
            b = a + 1
            c = a + n + 1
            d = a + n
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces.extend(
        [
            (b00, b01, b11),
            (b00, b11, b10),
            (t10, b10, b11),
            (t10, b11, t11),
            (t00, t01, b01),
            (t00, b01, b00),
            (t01, t11, b11),
            (t01, b11, b01),
            (t00, b00, b10),
            (t00, b10, t10),
        ]
    )
    indices = np.asarray(faces, dtype=np.int32).reshape(-1)
    return newton.Mesh(vertices, indices)
