"""Marker definitions shared by playback and project visualization code."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

type Pose = tuple[float, float, float, float, float, float, float]


def check_pose(pose: Pose) -> None:
    if len(pose) != 7 or not all(isfinite(value) for value in pose):
        raise ValueError("Marker poses must contain finite XYZ and XYZW values")
    if abs(sum(value * value for value in pose[3:]) - 1) > 1e-5:
        raise ValueError("Marker orientations must be unit quaternions")


@dataclass(frozen=True)
class Marker:
    """Draw a frame, a +X arrow, or a point in world or body coordinates."""

    name: str
    kind: Literal["frame", "arrow", "point"]
    pose: Pose
    size: float = 0.25
    color: int = 0xFFBA68
    group: str | None = None
    body: int | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        check_pose(self.pose)
        if self.kind not in ("frame", "arrow", "point"):
            raise ValueError("Unknown marker kind")
        if not isfinite(self.size) or self.size <= 0 or not 0 <= self.color <= 0xFFFFFF:
            raise ValueError("Markers require a positive size and an RGB color")
