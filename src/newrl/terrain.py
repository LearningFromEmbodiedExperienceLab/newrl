"""Static terrain geometry and optional spawn anchors for each replicated world."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import newton
import numpy as np
import warp as wp
from newton._src.utils.mesh import load_meshes_from_file

SpawnLocations = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Terrain:
    """Built terrain data available to MDPs in world coordinates."""

    spawn_locations: SpawnLocations = ()
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TerrainConfig:
    """Spawn locations are world-space ground anchors in meters.

    The MDP may sample an anchor and add it to the nominal root translation.
    Coordinates refer to the shared terrain seen by each Newton world.
    """

    spawn_locations: SpawnLocations = ()

    def __post_init__(self) -> None:
        for location in self.spawn_locations:
            if len(location) != 3 or not all(isfinite(value) for value in location):
                raise ValueError("Spawn locations must contain finite XYZ coordinates")

    def build(self, builder: newton.ModelBuilder) -> Terrain | SpawnLocations | None:
        """Add static shapes and optionally return terrain data or spawn anchors."""
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


@wp.kernel(enable_backward=False)
def _project_spawn_support(
    mesh: wp.uint64,
    origins: wp.array[wp.vec3],
    offsets: wp.array[wp.vec2],
    min_normal_z: float,
    max_drop: float,
    heights: wp.array2d[float],
) -> None:
    index, sample = wp.tid()
    origin = origins[index] + wp.vec3(offsets[sample][0], offsets[sample][1], 0.0)
    hit = wp.mesh_query_ray(mesh, origin, wp.vec3(0.0, 0.0, -1.0), max_drop)
    if hit.result and hit.normal[2] >= min_normal_z:
        heights[index, sample] = origin[2] - hit.t


@dataclass(frozen=True)
class ReconstructedTerrainConfig(TerrainConfig):
    """Load terrain.ply and terrain.json from a directory in a shared Z-up frame.

    Metadata origins are XYZ trajectory samples above the terrain, in meters.
    Support is sampled at the center and corners of a square around each origin.
    """

    directory: str = ""
    support_half_width: float = 0.45
    min_normal_z: float = 0.7
    max_height_span: float = 0.22
    max_drop: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all(
            isfinite(value) and value > 0
            for value in (self.support_half_width, self.max_height_span, self.max_drop)
        ):
            raise ValueError("Support half-width, height span, and drop must be finite and positive")
        if not 0 < self.min_normal_z <= 1:
            raise ValueError("Minimum support normal Z must be in (0, 1]")

    def build(self, builder: newton.ModelBuilder) -> Terrain:
        if not self.directory:
            raise ValueError("Set the reconstructed terrain directory")
        directory = Path(self.directory).expanduser()
        source = directory / "terrain.ply"
        metadata = (directory / "terrain.json").read_bytes()
        meta = json.loads(metadata)
        if not isinstance(meta, dict) or meta.get("up", [0, 0, 1]) != [0, 0, 1]:
            raise ValueError("terrain.json must contain a metadata object in a Z-up frame")

        first_shape = builder.shape_count
        MeshTerrainConfig(source=source).build(builder)
        if self.spawn_locations:
            return Terrain(self.spawn_locations, meta)
        origins = np.asarray(meta.get("origins"), dtype=np.float32)
        if (
            origins.ndim != 2
            or origins.shape[1] != 3
            or not len(origins)
            or not np.isfinite(origins).all()
        ):
            raise ValueError("terrain.json origins must be a nonempty array of finite XYZ points")

        key = sha256(source.read_bytes())
        key.update(metadata)
        # Increment the version when the support sampling or filtering changes.
        key.update(repr((
            1, self.support_half_width, self.min_normal_z, self.max_height_span, self.max_drop
        )).encode())
        cache = Path(".cache/newrl/terrain-spawns") / f"{key.hexdigest()}.json"
        if cache.is_file():
            return Terrain(tuple(map(tuple, json.loads(cache.read_text()))), meta)

        mesh = builder.shape_source[first_shape]
        with wp.ScopedDevice("cpu"):
            collision_mesh = wp.Mesh(
                points=wp.array(mesh.vertices, dtype=wp.vec3),
                indices=wp.array(mesh.indices, dtype=wp.int32),
            )
            w = self.support_half_width
            offsets = wp.array([(0, 0), (-w, -w), (-w, w), (w, -w), (w, w)], dtype=wp.vec2)
            heights = wp.full((len(origins), len(offsets)), np.nan, dtype=float)
            wp.launch(
                _project_spawn_support,
                dim=heights.shape,
                inputs=[
                    collision_mesh.id, wp.array(origins, dtype=wp.vec3), offsets,
                    self.min_normal_z, self.max_drop, heights,
                ],
            )
            support = heights.numpy()

        valid = np.isfinite(support).all(axis=1) & (
            np.ptp(support, axis=1) <= self.max_height_span
        )
        anchors = np.column_stack((origins[valid, :2], support[valid, 0]))
        if not len(anchors):
            raise ValueError(f"No suitable spawn supports found for {source}")
        locations = tuple(map(tuple, np.unique(anchors, axis=0).tolist()))
        cache.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as temp:
            json.dump(locations, temp)
        Path(temp.name).replace(cache)
        return Terrain(locations, meta)


@dataclass(frozen=True)
class FlatMeshTerrainConfig(TerrainConfig):
    """The flat slab from the mesh collision benchmark."""

    resolution: int = 1
    size: float = 20.0
    thickness: float = 1.0

    def build(self, builder: newton.ModelBuilder) -> None:
        mesh = flat_ground_mesh(self.resolution, self.size, self.thickness)
        builder.add_shape_mesh(body=-1, mesh=mesh, label="ground_mesh")


@dataclass(frozen=True)
class BorderConfig:
    """Rectangular terrain border, with dimensions in meters."""

    width: float = 0.25
    height: float = 1.5
    thickness: float = 0.25
    kind: str = "wall"

    def __post_init__(self) -> None:
        if self.kind not in ("wall", "apron"):
            raise ValueError("Border kind must be wall or apron")
        if not isfinite(self.width) or self.width < 0:
            raise ValueError("Border width must be finite and non-negative")
        if not isfinite(self.height) or self.height <= 0:
            raise ValueError("Border height must be finite and positive")
        if not isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("Border thickness must be finite and positive")


class TerrainTile(Protocol):
    def build(
        self,
        builder: newton.ModelBuilder,
        *,
        center: tuple[float, float],
        size: float,
        spawn_pad_width: float,
        rng: np.random.Generator,
        label: str,
    ) -> float:
        """Build a tile and return the ground height at its center."""
        ...


@dataclass(frozen=True)
class GeneratedTerrainConfig(TerrainConfig):
    """Repeat a fixed set of tile families in rows on one shared static map."""

    tiles: dict[str, TerrainTile]
    rows: int = 3
    tile_size: float = 8.0
    spawn_pad_width: float = 2.0
    seed: int = 0
    border: BorderConfig = field(default_factory=BorderConfig)
    # Retained for loading configurations saved before border presets.
    border_width: float | None = None
    border_height: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows < 1 or not self.tiles:
            raise ValueError("Generated terrain needs at least one row and tile family")
        if not 0.0 < self.spawn_pad_width < self.tile_size < float("inf"):
            raise ValueError("spawn_pad_width must be positive and smaller than tile_size")
        if self.seed < 0:
            raise ValueError("Terrain seed must be non-negative")
        if self.border_width is not None and (
            not isfinite(self.border_width) or self.border_width < 0
        ):
            raise ValueError("Border width must be finite and non-negative")
        if self.border_height is not None and (
            not isfinite(self.border_height) or self.border_height <= 0
        ):
            raise ValueError("Border height must be finite and positive")

    def build(self, builder: newton.ModelBuilder) -> SpawnLocations:
        rng = np.random.default_rng(self.seed)
        anchors = []
        for row in range(self.rows):
            x = (row + 0.5 - self.rows / 2) * self.tile_size
            for column, (name, tile) in enumerate(self.tiles.items()):
                y = (column + 0.5 - len(self.tiles) / 2) * self.tile_size
                z = tile.build(
                    builder,
                    center=(x, y),
                    size=self.tile_size,
                    spawn_pad_width=self.spawn_pad_width,
                    rng=rng,
                    label=f"terrain_{row}_{column}_{name}",
                )
                anchors.append((x, y, z))
        width = self.border.width if self.border_width is None else self.border_width
        height = self.border.height if self.border_height is None else self.border_height
        if width:
            half_x = self.rows * self.tile_size / 2
            half_y = len(self.tiles) * self.tile_size / 2
            for name, x, y, hx, hy in (
                ("west", -half_x - width / 2, 0.0, width / 2, half_y + width),
                ("east", half_x + width / 2, 0.0, width / 2, half_y + width),
                ("south", 0.0, -half_y - width / 2, half_x, width / 2),
                ("north", 0.0, half_y + width / 2, half_x, width / 2),
            ):
                top = height if self.border.kind == "wall" else 0.0
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(
                        (x, y, (top - self.border.thickness) / 2),
                        wp.quat_identity(),
                    ),
                    hx=hx,
                    hy=hy,
                    hz=(top + self.border.thickness) / 2,
                    label=f"border_{name}",
                )
        return self.spawn_locations or tuple(anchors)


def _add_ground_box(
    builder: newton.ModelBuilder,
    x: float,
    y: float,
    hx: float,
    hy: float,
    top: float,
    label: str,
) -> None:
    bottom = -0.25
    builder.add_shape_box(
        body=-1,
        xform=wp.transform((x, y, (top + bottom) / 2), wp.quat_identity()),
        hx=hx,
        hy=hy,
        hz=(top - bottom) / 2,
        label=label,
    )


@dataclass(frozen=True)
class FlatTileConfig:
    """Finite box slab with its upper surface at ground level."""

    def build(
        self,
        builder: newton.ModelBuilder,
        *,
        center: tuple[float, float],
        size: float,
        spawn_pad_width: float,
        rng: np.random.Generator,
        label: str,
    ) -> float:
        _add_ground_box(builder, *center, size / 2, size / 2, 0.0, label)
        return 0.0


@dataclass(frozen=True)
class RoughHeightfieldTileConfig:
    """Interpolated uniform noise with a flat spawn pad and level edges."""

    amplitude: float = 0.06
    sample_spacing: float = 0.1
    noise_spacing: float = 0.45
    gap: float = 0.01

    def __post_init__(self) -> None:
        if not isfinite(self.gap) or self.gap < 0:
            raise ValueError("Heightfield gap must be finite and non-negative")
        if not 0.0 <= self.amplitude < float("inf"):
            raise ValueError("Heightfield amplitude must be finite and non-negative")
        if not 0.0 < self.sample_spacing <= self.noise_spacing < float("inf"):
            raise ValueError("Spacings must satisfy 0 < sample_spacing <= noise_spacing")

    def build(
        self,
        builder: newton.ModelBuilder,
        *,
        center: tuple[float, float],
        size: float,
        spawn_pad_width: float,
        rng: np.random.Generator,
        label: str,
    ) -> float:
        half = size / 2
        count = ceil(size / self.sample_spacing) + 1
        axis = np.linspace(-half, half, count)
        noise_count = ceil(size / self.noise_spacing) + 1
        noise_axis = np.linspace(-half, half, noise_count)
        noise = rng.uniform(-self.amplitude, self.amplitude, (noise_count, noise_count))
        along_x = np.stack([np.interp(axis, noise_axis, row) for row in noise])
        heights = np.stack(
            [np.interp(axis, noise_axis, col) for col in along_x.T], axis=1
        )

        radius = np.maximum.outer(np.abs(axis), np.abs(axis))
        # Include the cells intersecting the pad so triangle interpolation keeps it flat.
        pad_edge = spawn_pad_width / 2 + size / (count - 1)
        heights *= np.clip((radius - pad_edge) / self.noise_spacing, 0.0, 1.0)
        heights *= np.clip((half - radius) / self.noise_spacing, 0.0, 1.0)
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.gap = self.gap
        builder.add_shape_heightfield(
            xform=wp.transform((*center, 0.0), wp.quat_identity()),
            heightfield=newton.Heightfield(heights, count, count, hx=half, hy=half),
            cfg=shape_cfg,
            label=label,
        )
        return 0.0


@dataclass(frozen=True)
class PyramidSlopeTileConfig:
    """Heightfield pyramid with level edges and a flat central platform."""

    slope: float = 0.25
    sample_spacing: float = 0.1
    inverted: bool = False
    gap: float = 0.01

    def __post_init__(self) -> None:
        if not isfinite(self.gap) or self.gap < 0:
            raise ValueError("Heightfield gap must be finite and non-negative")
        if not isfinite(self.slope) or self.slope <= 0:
            raise ValueError("Slope must be finite and positive")
        if not isfinite(self.sample_spacing) or self.sample_spacing <= 0:
            raise ValueError("Sample spacing must be finite and positive")

    def build(
        self,
        builder: newton.ModelBuilder,
        *,
        center: tuple[float, float],
        size: float,
        spawn_pad_width: float,
        rng: np.random.Generator,
        label: str,
    ) -> float:
        half = size / 2
        count = ceil(size / self.sample_spacing) + 1
        axis = np.linspace(-half, half, count)
        radius = np.maximum.outer(np.abs(axis), np.abs(axis))
        # Flatten every cell touching the spawn platform.
        pad_edge = min(half, spawn_pad_width / 2 + size / (count - 1))
        height = self.slope * (half - pad_edge)
        heights = self.slope * np.minimum(half - radius, half - pad_edge)
        if self.inverted:
            heights = -heights
            height = -height
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.gap = self.gap
        builder.add_shape_heightfield(
            xform=wp.transform((*center, 0.0), wp.quat_identity()),
            heightfield=newton.Heightfield(heights, count, count, hx=half, hy=half),
            cfg=shape_cfg,
            label=label,
        )
        return height


@dataclass(frozen=True)
class StairsTileConfig:
    """Square box treads ascending from a level rim to a flat center platform."""

    step_height: float = 0.10
    step_width: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.step_height < float("inf"):
            raise ValueError("step_height must be positive and finite")
        if not 0.0 < self.step_width < float("inf"):
            raise ValueError("step_width must be positive and finite")

    def build(
        self,
        builder: newton.ModelBuilder,
        *,
        center: tuple[float, float],
        size: float,
        spawn_pad_width: float,
        rng: np.random.Generator,
        label: str,
    ) -> float:
        x, y = center
        half = size / 2
        levels = int((size - spawn_pad_width) / (2 * self.step_width))
        if not levels:
            raise ValueError("Tile needs room for a tread outside the spawn pad")
        for level in range(levels):
            outer = half - level * self.step_width
            inner = outer - self.step_width
            offset = (outer + inner) / 2
            tread_half = self.step_width / 2
            top = level * self.step_height
            for side, (dx, dy, hx, hy) in enumerate(
                (
                    (-offset, 0.0, tread_half, outer),
                    (offset, 0.0, tread_half, outer),
                    (0.0, -offset, inner, tread_half),
                    (0.0, offset, inner, tread_half),
                )
            ):
                _add_ground_box(
                    builder, x + dx, y + dy, hx, hy, top, f"{label}_{level}_{side}"
                )

        platform_half = half - levels * self.step_width
        top = levels * self.step_height
        _add_ground_box(builder, x, y, platform_half, platform_half, top, f"{label}_pad")
        return top


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
