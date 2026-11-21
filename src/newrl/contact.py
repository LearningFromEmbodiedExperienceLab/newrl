"""Reduce solved terrain contacts to per-body sensor readings."""

from dataclasses import dataclass

import newton
import torch
import warp as wp
from tensordict import TensorDict

from .utils.string import resolve_scene_names


@dataclass(frozen=True)
class TerrainContactConfig:
    entity: str
    body_names: str | tuple[str, ...]
    load_threshold: float = 1.0  # N
    track_first_contact: bool = False
    track_air_time: bool = False
    track_contact_position: bool = False

    def __post_init__(self):
        if not 0.0 < self.load_threshold < float("inf"):
            raise ValueError("load_threshold must be positive and finite")


@wp.kernel
def _sum_terrain_forces(matrix: wp.array2d[wp.vec3], force: wp.array[wp.vec3]):
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
):
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


class TerrainContact:
    @property
    def fields(self):
        result = ["force_w", "loaded"]
        if self.track_history:
            result.append("first_contact")
        if self.config.track_air_time:
            result.extend(("current_air_time", "last_air_time"))
        if self.config.track_contact_position:
            result.extend(("last_contact_pos_w", "last_contact_pos_valid"))
        return tuple(result)

    @property
    def track_history(self):
        return (
            self.config.track_first_contact
            or self.config.track_air_time
            or self.config.track_contact_position
        )

    def __init__(self, config, entity):
        self.config = config
        indices = resolve_scene_names(config.body_names, entity.body_names)
        self.body_ids = entity.bodies[:, indices]
        self.sensor = None
        self.force = None

    def initialize(self, model, terrain_shapes):
        self.sensor = newton.sensors.SensorContact(
            model,
            sensing_bodies=self.body_ids.flatten().tolist(),
            counterpart_shapes=terrain_shapes,
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

    def force_view(self):
        return wp.to_torch(self.force).reshape(*self.body_ids.shape, 3)

    def checkpoint_views(self):
        views = {"force": self.force_view()}
        views.update(
            {
                name: wp.to_torch(value).reshape(*self.body_ids.shape, -1)
                for name, value in self.history.items()
            }
        )
        return views

    def reset(self, ids):
        for view in self.checkpoint_views().values():
            view[ids] = 0

    def begin_step(self):
        if self.track_history:
            self.history["first_contact"].zero_()

    def update(self, contacts, observables, state, dt):
        self.sensor.update(
            state if self.config.track_contact_position else None,
            contacts,
            observables=observables,
        )
        wp.launch(
            _sum_terrain_forces,
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

    def read(self, fields):
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
