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


class TerrainContact:
    fields = ("force_w", "loaded")

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

    def force_view(self):
        return wp.to_torch(self.force).reshape(*self.body_ids.shape, 3)

    def update(self, contacts, observables):
        self.sensor.update(None, contacts, observables=observables)
        wp.launch(
            _sum_terrain_forces,
            dim=self.force.size,
            inputs=[self.sensor.force_matrix],
            outputs=[self.force],
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
        return TensorDict(
            values, batch_size=self.body_ids.shape[:1], device=force.device
        )
