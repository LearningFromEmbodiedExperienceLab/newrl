"""Batched pinhole depth camera."""

import newton

from typing import TYPE_CHECKING

from dataclasses import dataclass
from math import isfinite, pi

import torch
import warp as wp
from newton.sensors import SensorCamera
from tensordict import TensorDict

from .camera import camera_refresh_mask

if TYPE_CHECKING:
    from newrl.scene import Scene
    from newrl.scene import RobotEntity

_MIN_FP16_DEPTH = wp.constant(2.0 ** -24)


@dataclass(frozen=True)
class DepthCameraConfig:
    """Pinhole camera with a body-relative mount in the forward-left-up frame.

    Position is in meters and roll, pitch, yaw are in radians about X, Y, Z.
    The camera looks along the mount's +X axis, with +Z up.
    Each world samples on reset and holds its frame for ``update_every``
    control steps between captures.
    Depth is stored and returned as FP16 meters, with zero for invalid pixels.
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

    def __post_init__(self) -> None:
        if not all(
            len(offset) == 3 and all(map(isfinite, offset))
            for offset in (self.pos, self.rpy)
        ):
            raise ValueError("Camera pos and rpy must be finite XYZ triples")
        if min(self.width, self.height) < 1 or not 0 < self.fov_y < pi:
            raise ValueError("Camera dimensions must be positive and fov_y must be in (0, pi)")
        if not 0 < self.max_depth <= 65504.0:
            raise ValueError("Camera max_depth must be positive and fit in FP16")
        if type(self.update_every) is not int or self.update_every < 1:
            raise ValueError("Camera update_every must be a positive integer")

    def build(self, scene: "Scene") -> "DepthCamera":
        return DepthCamera(self, scene.entities[self.entity], scene.model)


@wp.kernel(enable_backward=False)
def _camera_poses(
    body_q: wp.array[wp.transform],
    bodies: wp.array[wp.int32],
    mount: wp.transform,
    poses: wp.array[wp.transform],
) -> None:
    world = wp.tid()
    poses[world] = body_q[bodies[world]] * mount


@wp.kernel(enable_backward=False)
def _store_depth(
    rendered: wp.array3d[wp.float32],
    world_indices: wp.array[wp.int32],
    max_depth: float,
    depth: wp.array3d[wp.float16],
) -> None:
    world, row, col = wp.tid()
    if world_indices[world] < 0:
        return
    value = rendered[world, row, col]
    if value > 0.0 and value <= max_depth:
        # Reserve zero for invalid pixels even when a positive hit rounds down.
        depth[world, row, col] = wp.float16(wp.max(value, _MIN_FP16_DEPTH))
    else:
        depth[world, row, col] = wp.float16(0.0)


class DepthCamera:
    fields = ("depth", "valid")
    requires_bvh_refit = True

    def __init__(
        self,
        config: DepthCameraConfig,
        entity: "RobotEntity",
        model: newton.Model,
    ) -> None:
        self.config = config
        self.model = model
        (body,), _ = entity.find_bodies(config.body)
        self.body_ids = entity.bodies[:, body]
        self._bodies = wp.from_torch(self.body_ids.to(torch.int32), dtype=wp.int32)
        self.shape = torch.Size((config.height, config.width))
        # Convert Newton's camera axes to the mount's forward-left-up axes.
        optical = wp.quat(0.5, -0.5, -0.5, 0.5)
        rotation = wp.quat_rpy(*map(float, config.rpy))
        self.mount_transform = wp.transform(config.pos, rotation * optical)

    def initialize(self, scene: "Scene") -> None:
        config = self.config
        device = self.model.device
        worlds = self.model.world_count
        self._camera = SensorCamera(self.model, load_textures=False)
        self._poses = wp.empty(worlds, dtype=wp.transform, device=device)
        self._rays = self._camera.compute_camera_rays_pinhole(
            config.width, config.height, camera_fov=config.fov_y, device=device
        )
        self._depth = self._camera.allocate_image(worlds, config.width, config.height, wp.float16)
        self._depth_view = wp.to_torch(self._depth)
        self._world_indices = torch.arange(worlds, dtype=torch.int32, device=self._depth_view.device)
        self._steps_until_refresh = (
            self._world_indices.new_full((worlds,), config.update_every, dtype=torch.int64)
            if config.update_every > 1
            else None
        )

    def sample(
        self,
        state: newton.State,
        *,
        reset: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> None:
        """Refresh due worlds or the worlds selected for an immediate reset frame."""
        due = camera_refresh_mask(
            self._steps_until_refresh, self.config.update_every,
            reset=reset, reset_mask=reset_mask,
        )

        world_indices = None
        if due is not None:
            selected = torch.where(
                due, self._world_indices, int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE)
            )
            world_indices = wp.from_torch(selected, dtype=wp.int32)
        wp.launch(
            _camera_poses, dim=self.model.world_count,
            inputs=[state.body_q, self._bodies, self.mount_transform, self._poses],
            device=self.model.device,
        )
        # Newton renders FP32. Keep the staging image local to the capture.
        rendered = wp.from_torch(torch.empty_like(self._depth_view, dtype=torch.float32))
        self._camera.update(
            state, self._poses, self._rays,
            forward_depth_image=rendered, world_indices=world_indices,
        )
        wp.launch(
            _store_depth, dim=self._depth.shape,
            inputs=[
                rendered,
                wp.from_torch(self._world_indices) if world_indices is None else world_indices,
                self.config.max_depth,
                self._depth,
            ],
            device=self.model.device,
        )

    def checkpoint_views(self) -> dict[str, torch.Tensor]:
        views = {"depth": self._depth_view}
        if self._steps_until_refresh is not None:
            views["steps_until_refresh"] = self._steps_until_refresh.unsqueeze(-1)
        return views

    def depth_frame(self, world: int) -> torch.Tensor:
        """Return a view of one world's latest cached depth in meters."""
        return self._depth_view[world]

    def read(self, state: newton.State, fields: tuple[str, ...]) -> TensorDict:
        values = {}
        if "depth" in fields:
            values["depth"] = self._depth_view.clone()
        if "valid" in fields:
            values["valid"] = self._depth_view > 0
        return TensorDict(values, batch_size=[self.model.world_count])
