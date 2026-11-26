"""Resolve a portable gsrender scene to a cloud and its simulation transform."""

import json
from math import isclose, isfinite
from pathlib import Path

IDENTITY_POSE = (0., 0., 0., 0., 0., 0., 1.)


def resolve_gaussian_scene(
    source: str, pose: tuple[float, ...] = IDENTITY_POSE, *, browser: bool = False
) -> tuple[str, tuple[float, ...]]:
    manifest = Path(source).expanduser()
    if manifest.suffix.lower() != ".json":
        return source, pose
    if pose != IDENTITY_POSE:
        raise ValueError("Set the cloud transform in scene.json when using a scene manifest")
    root = manifest.resolve().parent
    data = json.loads(manifest.read_text())
    if (data.get("version"), data.get("coordinateSystem"), data.get("units")) != (1, "z-up", "meters"):
        raise ValueError("Expected a version 1 Z-up scene in meters")
    transform = tuple(data["worldFromSource"])
    if len(transform) != 7 or not all(map(isfinite, transform)):
        raise ValueError("worldFromSource must contain finite XYZ and XYZW values")
    if not isclose(sum(value * value for value in transform[3:]), 1, abs_tol=1e-6):
        raise ValueError("worldFromSource quaternion must be unit length")
    cloud = data.get("cloud")
    supported = (".ply", ".spz") if browser else (".ply",)
    if not browser or not cloud or Path(cloud).suffix.lower() not in supported:
        cloud = data.get("renderCloud") or cloud
    if not cloud or Path(cloud).is_absolute() or Path(cloud).suffix.lower() not in supported:
        raise ValueError("Scene needs a relative renderCloud PLY or a supported cloud")
    path = (root / cloud).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Scene cloud is missing or outside the bundle: {cloud}")
    return str(path), transform
