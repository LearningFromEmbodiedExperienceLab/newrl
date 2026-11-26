"""Renderer-independent task arrows and the browser playback adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self, TYPE_CHECKING

from .markers import Marker

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torch import Tensor

    from newrl.env import StatefulEnvironment
    from newrl.scene import Scene
    from .browser import BrowserViewer


class TaskVisualization(Protocol):
    """Sample world-space arrow origins and vectors from read-only task state."""

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]: ...

    @property
    def mdp_fields(self) -> tuple[str, ...]: ...

    @property
    def arrows(self) -> tuple[Marker, ...]: ...

    def _initialize(self, scene: Scene) -> Self: ...

    def sample(self, physical_state: TensorDict, mdp_state: TensorDict) -> Mapping[str, Tensor]:
        """Return [num_envs, 6] XYZ origin and XYZ vector for each arrow."""
        ...


class TaskVisualizationPlayback:
    """Read task marker inputs only when publishing a viewport frame."""

    def __init__(self, env: StatefulEnvironment, viewer: BrowserViewer) -> None:
        self.env = env
        self.viewer = viewer
        self.components = tuple(term._initialize(env.scene) for term in env.mdp.visualizations())
        self.field_plan = env.scene.resolve_physical_fields(
            set().union(*(term.physical_fields for term in self.components))
        )
        for name, fields in self.field_plan.items():
            if name in env.scene.sensors and name not in env.scene.physical_read_plan and name not in env.scene.cameras:
                raise ValueError(f"Task visualization requires an active sensor: {name}")
            if "joint_torque" in fields and not env.scene.needs_torque:
                raise ValueError("Task visualization requires joint torque in the environment read plan")
        for term in self.components:
            for arrow in term.arrows:
                self.viewer.add_marker(arrow)

    def update(self) -> None:
        if not self.components:
            return
        env = self.env
        physical = env.scene.read(env.state, env.observables, field_plan=self.field_plan)
        for term in self.components:
            for name, vectors in term.sample(physical, env.mdp_state.select(*term.mdp_fields)).items():
                self.viewer.set_marker_vectors(name, vectors)
