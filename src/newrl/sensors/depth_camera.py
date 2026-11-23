"""Batched pinhole depth camera."""

from dataclasses import dataclass
from math import isfinite, pi

import torch
import warp as wp
from newton.sensors import SensorCamera
from tensordict import TensorDict

from ..utils.string import resolve_scene_names


@dataclass(frozen=True)
class DepthCameraConfig:
    """Pinhole camera with a body-relative mount in the forward-left-up frame.

    Position is in meters and roll, pitch, yaw are in radians about X, Y, Z.
    The camera looks along the mount's +X axis, with +Z up.
    Each world samples on reset and holds its frame for ``update_every``
    control steps between captures.
    """

    entity: str = "robot"
    body: str = "base_link"
    pos: tuple[float, float, float] = (0.3, 0.0, 0.1)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: int = 64
    height: int = 48
    fov_y: float = pi / 3
    max_depth: float = 8.0
    update_every: int = 1  # Control steps between frames.

    def __post_init__(self):
        if not all(
            len(offset) == 3 and all(map(isfinite, offset))
            for offset in (self.pos, self.rpy)
        ):
            raise ValueError("Camera pos and rpy must be finite XYZ triples")
        if min(self.width, self.height) < 1 or not 0 < self.fov_y < pi:
            raise ValueError("Camera dimensions must be positive and fov_y must be in (0, pi)")
        if not isfinite(self.max_depth) or self.max_depth <= 0:
            raise ValueError("Camera max_depth must be finite and positive")
        if type(self.update_every) is not int or self.update_every < 1:
            raise ValueError("Camera update_every must be a positive integer")

    def build(self, scene):
        return DepthCamera(self, scene.entities[self.entity], scene.model)


@wp.kernel(enable_backward=False)
def _camera_poses(
    body_q: wp.array[wp.transform],
    bodies: wp.array[wp.int32],
    mount: wp.transform,
    poses: wp.array[wp.transform],
):
    world = wp.tid()
    poses[world] = body_q[bodies[world]] * mount


class DepthCamera:
    fields = ("depth", "valid")

    def __init__(self, config, entity, model):
        self.config = config
        self.model = model
        (body,) = resolve_scene_names(config.body, entity.body_names)
        self._bodies = wp.from_torch(entity.bodies[:, body].to(torch.int32), dtype=wp.int32)
        self.shape = torch.Size((config.height, config.width))
        # Convert Newton's camera axes to the mount's forward-left-up axes.
        optical = wp.quat(0.5, -0.5, -0.5, 0.5)
        rotation = wp.quat_rpy(*map(float, config.rpy))
        self._mount = wp.transform(config.pos, rotation * optical)

    def initialize(self, scene):
        config = self.config
        device = self.model.device
        worlds = self.model.world_count
        self._camera = SensorCamera(self.model, load_textures=False)
        self._poses = wp.empty(worlds, dtype=wp.transform, device=device)
        self._rays = self._camera.compute_camera_rays_pinhole(
            config.width, config.height, camera_fov=config.fov_y, device=device
        )
        self._depth = self._camera.allocate_image(worlds, config.width, config.height, wp.float32)
        self._depth_view = wp.to_torch(self._depth)
        self._world_indices = torch.arange(worlds, dtype=torch.int32, device=self._depth_view.device)
        self._steps_until_refresh = (
            self._world_indices.new_full((worlds,), config.update_every, dtype=torch.int64)
            if config.update_every > 1
            else None
        )

    def sample(self, state, *, reset=False, reset_mask=None):
        """Refresh due worlds or the worlds selected for an immediate reset frame."""
        remaining = self._steps_until_refresh
        due = reset_mask
        if remaining is not None:
            if reset:
                if reset_mask is None:
                    remaining.fill_(self.config.update_every)
                else:
                    remaining.masked_fill_(reset_mask, self.config.update_every)
            else:
                remaining.sub_(1)
                due = remaining == 0
                remaining.masked_fill_(due, self.config.update_every)

        world_indices = None
        if due is not None:
            selected = torch.where(
                due, self._world_indices, int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE)
            )
            world_indices = wp.from_torch(selected, dtype=wp.int32)
        wp.launch(
            _camera_poses, dim=self.model.world_count,
            inputs=[state.body_q, self._bodies, self._mount, self._poses],
            device=self.model.device,
        )
        self._camera.update(
            state, self._poses, self._rays,
            forward_depth_image=self._depth, world_indices=world_indices,
        )

    def checkpoint_views(self):
        views = {"depth": self._depth_view}
        if self._steps_until_refresh is not None:
            views["steps_until_refresh"] = self._steps_until_refresh.unsqueeze(-1)
        return views

    def read(self, state, fields):
        depth = self._depth_view
        valid = (depth > 0.0) & (depth <= self.config.max_depth)
        values = {"valid": valid}
        if "depth" in fields:
            values["depth"] = torch.where(valid, depth, 0.0)
        return TensorDict({name: values[name] for name in fields}, batch_size=[self.model.world_count])
