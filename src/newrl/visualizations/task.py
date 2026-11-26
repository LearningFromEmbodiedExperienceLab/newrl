"""Renderer-independent task and sensor markers and the browser playback adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self, TYPE_CHECKING

from .markers import Marker
from newrl.sensors.contact import TerrainContact

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torch import Tensor

    from newrl.env import StatefulEnvironment
    from newrl.scene import Scene
    from .browser import BrowserViewer


class TaskVisualization(Protocol):
    """Sample world-space marker data from read-only task and sensor state."""

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]: ...

    @property
    def mdp_fields(self) -> tuple[str, ...]: ...

    @property
    def markers(self) -> tuple[Marker, ...]: ...

    def _initialize(self, scene: Scene) -> Self: ...

    def sample(self, physical_state: TensorDict, mdp_state: TensorDict) -> Mapping[str, Tensor | tuple[Tensor, Tensor]]:
        """Return arrow origin/vector rows or point positions and validity masks."""
        ...


class TaskVisualizationPlayback:
    """Read task marker inputs only when publishing a viewport frame."""

    def __init__(self, env: StatefulEnvironment, viewer: BrowserViewer) -> None:
        self.env = env
        self.viewer = viewer
        components = list(env.mdp.visualizations())
        active_sensors = set(env.scene.physical_read_plan) | set(env.scene.cameras)
        with env._stream():
            for name, sensor in env.scene.sensors.items():
                visualize = getattr(sensor, "visualizations", None)
                if visualize is None:
                    continue
                sensor_components = visualize(name)
                if not sensor_components:
                    continue
                if name not in active_sensors:
                    if isinstance(sensor, TerrainContact):
                        raise ValueError(f"Contact visualization requires solver contact tracking for {name}")
                    sensor.initialize(env.scene)
                    active_sensors.add(name)
                components.extend(sensor_components)
            self.components = tuple(term._initialize(env.scene) for term in components)
        self.field_plan = env.scene.resolve_physical_fields(
            set().union(*(term.physical_fields for term in self.components))
        )
        for name, fields in self.field_plan.items():
            if name in env.scene.sensors and name not in active_sensors:
                raise ValueError(f"Task visualization requires an active sensor: {name}")
            if "joint_torque" in fields and not env.scene.needs_torque:
                raise ValueError("Task visualization requires joint torque in the environment read plan")
        for term in self.components:
            for marker in term.markers:
                self.viewer.add_marker(marker)

    def update(self) -> None:
        if not self.components:
            return
        env = self.env
        with env._stream():
            physical = env.scene.read(env.state, env.observables, field_plan=self.field_plan)
            for term in self.components:
                for name, sample in term.sample(physical, env.mdp_state.select(*term.mdp_fields)).items():
                    if isinstance(sample, tuple):
                        self.viewer.set_marker_points(name, *sample)
                    else:
                        self.viewer.set_marker_vectors(name, sample)
