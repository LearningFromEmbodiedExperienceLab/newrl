"""Batched downward raycasts against the shared terrain."""

from dataclasses import dataclass
from math import isfinite

import newton
import torch
import warp as wp
from tensordict import TensorDict

from ..utils.string import resolve_scene_names


@dataclass(frozen=True)
class HeightMapConfig:
    """Yaw-aligned height grid with forward rows and lateral columns.

    Heights are relative to the mounting body's Z position. All lengths are
    in meters. The center offset moves the grid forward from the body.
    """

    entity: str = "robot"
    body: str = "base_link"
    rows: int = 21
    cols: int = 15
    spacing: float = 0.1
    forward_offset: float = 0.7
    ray_start_height: float = 0.5
    ray_length: float = 3.0

    def __post_init__(self):
        if min(self.rows, self.cols) < 1:
            raise ValueError("Height map dimensions must be positive")
        distances = (self.spacing, self.forward_offset, self.ray_start_height, self.ray_length)
        if not all(map(isfinite, distances)):
            raise ValueError("Height map distances must be finite")
        if self.spacing <= 0 or self.ray_start_height < 0 or self.ray_length <= 0:
            raise ValueError("Height map spacing and ray length must be positive and ray start nonnegative")

    def build(self, scene):
        return HeightMap(self, scene.entities[self.entity], scene.model)


@wp.kernel(enable_backward=False)
def _height_ray_origins(
    body_q: wp.array[wp.transform],
    bodies: wp.array[wp.int32],
    rows: int,
    cols: int,
    spacing: float,
    forward_offset: float,
    ray_start_height: float,
    origins: wp.array[wp.vec3],
):
    world, row, col = wp.tid()
    pose = body_q[bodies[world]]
    q = wp.transform_get_rotation(pose)
    yaw = wp.atan2(
        2.0 * (q[3] * q[2] + q[0] * q[1]),
        1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]),
    )
    x = forward_offset + (0.5 * float(rows - 1) - float(row)) * spacing
    y = (0.5 * float(cols - 1) - float(col)) * spacing
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    offset = wp.vec3(c * x - s * y, s * x + c * y, ray_start_height)
    origins[(world * rows + row) * cols + col] = wp.transform_get_translation(pose) + offset


class HeightMap:
    fields = ("height", "valid")

    def __init__(self, config, entity, model):
        self.config = config
        self.model = model
        (body,) = resolve_scene_names(config.body, entity.body_names)
        self._bodies = wp.from_torch(entity.bodies[:, body].to(torch.int32), dtype=wp.int32)
        self.shape = torch.Size((config.rows, config.cols))

    def initialize(self, scene):
        device = self.model.device
        count = self.model.world_count * self.shape.numel()
        self._origins = wp.empty(count, dtype=wp.vec3, device=device)
        self._directions = wp.full(count, wp.vec3(0, 0, -1), dtype=wp.vec3, device=device)
        # NewRL adds the shared terrain to the global world.
        self._worlds = wp.full(count, -1, dtype=wp.int32, device=device)
        self._distances = wp.empty(count, dtype=wp.float32, device=device)

    def read(self, state, fields):
        config = self.config
        worlds = self.model.world_count
        device = self.model.device
        wp.launch(
            _height_ray_origins, dim=(worlds, *self.shape),
            inputs=[
                state.body_q, self._bodies, config.rows, config.cols, config.spacing,
                config.forward_offset, config.ray_start_height, self._origins,
            ],
            device=device,
        )
        newton.intersect_ray(
            self.model, ray_origins=self._origins, ray_directions=self._directions,
            ray_worlds=self._worlds, out_dist=self._distances,
        )
        distance = wp.to_torch(self._distances).reshape(worlds, *self.shape)
        valid = (distance >= 0.0) & (distance <= config.ray_length)
        values = {"valid": valid}
        if "height" in fields:
            values["height"] = torch.where(valid, config.ray_start_height - distance, 0.0)
        return TensorDict({name: values[name] for name in fields}, batch_size=[worlds])
