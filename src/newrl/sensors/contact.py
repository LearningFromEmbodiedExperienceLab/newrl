"""Reduce solved contacts to per-body sensor readings."""

from typing import TYPE_CHECKING

from dataclasses import dataclass

import newton
import torch
import warp as wp
from tensordict import TensorDict

if TYPE_CHECKING:
    from newrl.scene import Entity, Scene
    from newrl.visualizations.task import TaskVisualization



@dataclass(frozen=True)
class ContactSensorConfig:
    entity: str
    body_names: str | tuple[str, ...]
    load_threshold: float = 1.0  # N
    track_first_contact: bool = False
    track_air_time: bool = False
    track_contact_position: bool = False

    counterpart: str | None = None  # None selects terrain shapes.

    def __post_init__(self) -> None:
        if not 0.0 < self.load_threshold < float("inf"):
            raise ValueError("load_threshold must be positive and finite")

    def build(self, scene: "Scene") -> "ContactSensor":
        return ContactSensor(self, scene.entities[self.entity])


@wp.kernel
def _sum_forces(matrix: wp.array2d[wp.vec3], force: wp.array[wp.vec3]) -> None:
    row = wp.tid()
    total = wp.vec3(0.0)
    for column in range(matrix.shape[1]):
        total += matrix[row, column]
    force[row] = total


@wp.kernel
def _update_contact_history(
    force: wp.array[wp.vec3],
    positions: wp.array2d[wp.vec3],
    weights: wp.array2d[wp.float32],
    threshold: float,
    dt: float,
    track_air: bool,
    track_position: bool,
    previous: wp.array[wp.bool],
    first: wp.array[wp.bool],
    current_air: wp.array[wp.float32],
    last_air: wp.array[wp.float32],
    last_position: wp.array[wp.vec3],
    valid: wp.array[wp.bool],
) -> None:
    i = wp.tid()
    loaded = wp.length(force[i]) > threshold
    touchdown = loaded and not previous[i]
    if touchdown:
        first[i] = True
        if track_air:
            last_air[i] = current_air[i]
        if track_position:
            weighted = wp.vec3(0.0)
            total = float(0.0)  # noqa: UP018 - mutable Warp variable
            for j in range(positions.shape[1]):
                w = weights[i, j]
                weighted += positions[i, j] * w
                total += w
            if total > 0.0:
                last_position[i] = weighted / total
                valid[i] = True
    if track_air:
        if loaded:
            current_air[i] = 0.0
        else:
            current_air[i] += dt
    previous[i] = loaded


class ContactSensor:
    @property
    def fields(self) -> tuple[str, ...]:
        result = ["force_w", "loaded"]
        if self.track_history:
            result.append("first_contact")
        if self.config.track_air_time:
            result.extend(("current_air_time", "last_air_time"))
        if self.config.track_contact_position:
            result.extend(("last_contact_pos_w", "last_contact_pos_valid"))
        return tuple(result)

    @property
    def track_history(self) -> bool:
        return (
            self.config.track_first_contact
            or self.config.track_air_time
            or self.config.track_contact_position
        )

    def __init__(self, config: ContactSensorConfig, entity: "Entity") -> None:
        self.config = config
        indices, _ = entity.find_bodies(config.body_names)
        self.body_ids = entity.body_indices[:, indices]

    def visualizations(self, name: str) -> tuple["TaskVisualization", ...]:
        from newrl.visualizations.contact import ContactPositions

        return (ContactPositions(name),) if self.config.track_contact_position else ()

    def initialize(self, scene: "Scene") -> None:
        model = scene.model
        counterpart = self.config.counterpart
        if counterpart is None:
            targets = {"counterpart_shapes": scene.terrain_shapes}
        else:
            if counterpart not in scene.entities:
                raise ValueError(f"Unknown contact counterpart {counterpart!r}")
            targets = {"counterpart_bodies": scene.entities[counterpart].body_indices.flatten().tolist()}
        self.sensor = newton.sensors.SensorContact(
            model,
            sensing_bodies=self.body_ids.flatten().tolist(),
            **targets,
            measure_total=False,
        )
        self.force = wp.zeros(self.body_ids.numel(), dtype=wp.vec3, device=model.device)
        self.history = {}
        if self.track_history:
            for name, dtype in (
                ("previous_loaded", wp.bool),
                ("first_contact", wp.bool),
                ("current_air_time", wp.float32),
                ("last_air_time", wp.float32),
                ("last_contact_pos_w", wp.vec3),
                ("last_contact_pos_valid", wp.bool),
            ):
                self.history[name] = wp.zeros(
                    self.body_ids.numel(), dtype=dtype, device=model.device
                )

    def force_view(self) -> torch.Tensor:
        return wp.to_torch(self.force).reshape(*self.body_ids.shape, 3)

    def checkpoint_views(self) -> dict[str, torch.Tensor]:
        views = {"force": self.force_view()}
        views.update(
            {
                name: wp.to_torch(value).reshape(*self.body_ids.shape, -1)
                for name, value in self.history.items()
            }
        )
        return views

    def reset(self, ids: torch.Tensor | slice) -> None:
        for view in self.checkpoint_views().values():
            view[ids] = 0

    def begin_step(self) -> None:
        if self.track_history:
            self.history["first_contact"].zero_()

    def update(
        self,
        contacts: newton.Contacts | None,
        observables: newton.solvers.SolverMuJoCo.Observables | newton.solvers.SolverSteel.Observables | None,
        state: newton.State,
        dt: float,
    ) -> None:
        self.sensor.update(
            state if self.config.track_contact_position else None,
            contacts,
            observables=observables,
        )
        wp.launch(
            _sum_forces,
            dim=self.force.size,
            inputs=[self.sensor.force_matrix],
            outputs=[self.force],
            device=self.force.device,
        )
        if self.track_history:
            wp.launch(
                _update_contact_history,
                dim=self.force.size,
                inputs=[
                    self.force,
                    self.sensor.position_matrix,
                    self.sensor._position_weight,
                    self.config.load_threshold,
                    dt,
                    self.config.track_air_time,
                    self.config.track_contact_position,
                    *self.history.values(),
                ],
                device=self.force.device,
            )

    def read(self, state: newton.State, fields: tuple[str, ...]) -> TensorDict:
        force = self.force_view()
        values = {}
        if "force_w" in fields:
            values["force_w"] = force.clone()
        if "loaded" in fields:
            values["loaded"] = (
                torch.linalg.vector_norm(force, dim=-1) > self.config.load_threshold
            )
        history_views = self.checkpoint_views() if self.history else {}
        for name in fields:
            if name in self.history and name != "previous_loaded":
                view = history_views[name]
                values[name] = (
                    view if name == "last_contact_pos_w" else view.squeeze(-1)
                ).clone()
        return TensorDict(
            values, batch_size=self.body_ids.shape[:1], device=force.device
        )
