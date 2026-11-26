"""Viewer presets and setup shared by the playback commands."""

from __future__ import annotations

from dataclasses import dataclass
from socket import socket
from typing import TYPE_CHECKING

import newton

from newrl.terrains import PlaneTerrainConfig, Terrain
from newrl.utils.config import cs

if TYPE_CHECKING:
    from .browser import BrowserViewer
    from newrl.env import StatefulEnvironment
    from .timeline import Timeline


def viewer_world_offsets(terrain: Terrain | None, num_envs: int) -> tuple[float, float, float]:
    """Viewing offset applied on top of each world's simulated pose.

    Every world shares one static terrain in world coordinates. ``set_model``
    also spreads worlds by their own size. Either offset draws a robot over a
    different point on that terrain than the one it contacts, so it appears to
    float or sink wherever the ground height changes. An infinite plane is the
    same height everywhere, so multiple worlds can be separated visually.
    """
    if num_envs > 1 and (terrain == "plane" or isinstance(terrain, PlaneTerrainConfig)):
        return (2.0, 2.0, 0.0)
    return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GLViewerConfig:
    def build(self, env: StatefulEnvironment) -> newton.viewer.ViewerGL:
        return newton.viewer.ViewerGL()


@dataclass(frozen=True)
class ViserViewerConfig:
    host: str = "127.0.0.1"
    port: int = 8081

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError("viewer.port must be between 0 and 65535")

    def build(self, env: StatefulEnvironment) -> newton.viewer.ViewerViser:
        port = viewer_port_or_ephemeral(self.host, self.port)
        return newton.viewer.ViewerViser(host=self.host, port=port)


@dataclass(frozen=True)
class BrowserViewerConfig:
    port: int = 8081
    splat_path: str | None = None
    recording_dir: str = "recordings"
    terrain_origin_frames: bool = True
    camera_preview: bool = False
    task_visualizations: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError("viewer.port must be between 0 and 65535")

    def build(self, env: StatefulEnvironment, *, timeline: Timeline | None = None) -> BrowserViewer:
        from .browser import BrowserViewer

        if self.camera_preview:
            with env._stream():
                env.scene.enable_camera_previews()
        return BrowserViewer(
            env.scene, self.port, self.splat_path,
            recording_dir=self.recording_dir, task_name=env.config.name,
            timeline=timeline,
            terrain_origin_frames=self.terrain_origin_frames,
        )


cs.store("gl", GLViewerConfig(), group="viewer")
cs.store("viser", ViserViewerConfig(), group="viewer")
cs.store("browser", BrowserViewerConfig(), group="viewer")


def viewer_port_or_ephemeral(host: str, port: int) -> int:
    """Use the requested port when free, otherwise ask the OS for a free one."""
    if port == 0:
        return 0
    try:
        with socket() as probe:
            probe.bind((host, port))
    except OSError:
        print(
            f"Viewer port {port} is unavailable. Requesting an available port.",
            flush=True,
        )
        return 0
    return port


