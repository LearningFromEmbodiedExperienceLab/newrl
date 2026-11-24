"""Body-mounted RGB and depth camera for a shared static Gaussian splat scene."""

from dataclasses import dataclass
from math import isfinite, pi, tan
from typing import Literal

import torch
import warp as wp
from tensordict import TensorDict

from ..utils.string import resolve_scene_names
from .camera import camera_refresh_mask


@dataclass(frozen=True)
class GaussianCameraConfig:
    """Configure an optional gsrender camera with the same mount axes as depth.

    ``world_from_cloud`` is the cloud-to-simulation rigid transform as XYZ
    translation in meters followed by an XYZW quaternion. Cloud units are meters.
    Camera mount position and roll, pitch, yaw use meters and radians.
    """

    source: str
    world_from_cloud: tuple[float, float, float, float, float, float, float] = (
        0., 0., 0., 0., 0., 0., 1.,
    )
    entity: str = "robot"
    body: str = "base_link"
    pos: tuple[float, float, float] = (0.3, 0.0, 0.1)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: int = 64
    height: int = 48
    fov_y: float = pi / 3
    near: float = 0.05
    far: float = 8.0
    update_every: int = 5
    cell_size: float = 12.0
    cache_cells: int = 12
    batch_size: int = 32
    radius_clip: float = 0.0
    mode: Literal["rgb", "depth", "rgbd"] = "rgb"
    min_alpha: float = 0.1

    def __post_init__(self):
        if min(self.width, self.height) < 1 or not 0 < self.fov_y < pi:
            raise ValueError("Camera dimensions and vertical field of view must be positive")
        if not 0 < self.near < self.far < float("inf"):
            raise ValueError("Camera near and far planes must be finite and ordered")
        if self.mode not in ("rgb", "depth", "rgbd"):
            raise ValueError("Gaussian camera mode must be rgb, depth, or rgbd")
        if self.mode != "rgb" and (self.far > 65504.0 or not 0 < self.min_alpha <= 1):
            raise ValueError("Gaussian depth needs an FP16 far plane and min_alpha in (0, 1]")
        if type(self.update_every) is not int or self.update_every < 1:
            raise ValueError("Camera update_every must be a positive integer")
        for values, size in ((self.pos, 3), (self.rpy, 3), (self.world_from_cloud, 7)):
            if len(values) != size or not all(map(isfinite, values)):
                raise ValueError("Camera mount and cloud transform must be finite")
        if abs(sum(x*x for x in self.world_from_cloud[3:]) - 1.0) > 1e-5:
            raise ValueError("Cloud transform must use a unit quaternion")

    def build(self, scene):
        return GaussianCamera(self, scene.entities[self.entity], scene.model)


@wp.kernel(enable_backward=False)
def _view_matrices(
    poses: wp.array[wp.transform],
    bodies: wp.array[wp.int32],
    mount: wp.transform,
    cloud_from_world: wp.transform,
    views: wp.array[wp.mat44],
):
    world = wp.tid()
    view = wp.transform_inverse(cloud_from_world * poses[bodies[world]] * mount)
    rot = wp.quat_to_matrix(wp.transform_get_rotation(view))
    pos = wp.transform_get_translation(view)
    views[world] = wp.mat44(
        rot[0, 0], rot[0, 1], rot[0, 2], pos[0],
        rot[1, 0], rot[1, 1], rot[1, 2], pos[1],
        rot[2, 0], rot[2, 1], rot[2, 2], pos[2],
        0.0, 0.0, 0.0, 1.0,
    )


class GaussianCamera:
    requires_bvh_refit = False

    def __init__(self, config, entity, model):
        self.config = config
        self.fields = {
            "rgb": ("rgb",), "depth": ("depth", "valid"), "rgbd": ("rgb", "depth", "valid")
        }[config.mode]
        self.model = model
        (body,) = resolve_scene_names(config.body, entity.body_names)
        self.body_ids = entity.bodies[:, body]
        self._bodies = wp.from_torch(self.body_ids.to(torch.int32))
        self.shape = torch.Size((config.height, config.width))
        rotation = wp.quat_rpy(*map(float, config.rpy))
        # Frustum display uses the same OpenGL frame as the depth camera.
        self.mount_transform = wp.transform(config.pos, rotation * wp.quat(0.5, -0.5, -0.5, 0.5))
        self._cv_mount = self.mount_transform * wp.transform(
            wp.vec3(0.0), wp.quat(1.0, 0.0, 0.0, 0.0)
        )
        transform = config.world_from_cloud
        self._cloud_from_world = wp.transform_inverse(
            wp.transform(wp.vec3(*transform[:3]), wp.quat(*transform[3:]))
        )

    def initialize(self, scene):
        try:
            from gsrender.live import LiveGaussianRenderer
        except ImportError as error:
            raise ImportError("Install gsrender[gsplat] to enable Gaussian camera frames") from error

        config = self.config
        device = torch.device(wp.device_to_torch(self.model.device))
        worlds = self.model.world_count
        self._renderer = LiveGaussianRenderer(
            config.source, device, cell_size=config.cell_size, cache_cells=config.cache_cells,
            batch_size=config.batch_size, radius_clip=config.radius_clip,
        )
        self._views = wp.empty(worlds, dtype=wp.mat44, device=self.model.device)
        self._view_tensor = wp.to_torch(self._views)
        self._rgb = (
            torch.zeros((worlds, *self.shape, 3), dtype=torch.uint8, device=device)
            if "rgb" in self.fields else None
        )
        self._depth = (
            torch.zeros((worlds, *self.shape), dtype=torch.float16, device=device)
            if "depth" in self.fields else None
        )
        focal = 0.5 * config.height / tan(0.5 * config.fov_y)
        self._intrinsics = torch.tensor(
            ((focal, 0, config.width / 2), (0, focal, config.height / 2), (0, 0, 1)),
            dtype=torch.float32, device=device,
        )
        self._steps_until_refresh = (
            torch.full((worlds,), config.update_every, dtype=torch.int64, device=device)
            if config.update_every > 1 else None
        )

    def sample(self, state, *, reset=False, reset_mask=None):
        config = self.config
        due = camera_refresh_mask(
            self._steps_until_refresh, config.update_every, reset=reset, reset_mask=reset_mask,
        )
        wp.launch(
            _view_matrices, dim=self.model.world_count, device=self.model.device,
            inputs=[state.body_q, self._bodies, self._cv_mount, self._cloud_from_world, self._views],
        )
        self._renderer.render(
            self._view_tensor, self._intrinsics, near=config.near, far=config.far,
            rgb_out=self._rgb, depth_out=self._depth, min_alpha=config.min_alpha, active=due,
        )

    def checkpoint_views(self):
        views = {}
        if self._rgb is not None:
            views["rgb"] = self._rgb
        if self._depth is not None:
            views["depth"] = self._depth
        if self._steps_until_refresh is not None:
            views["steps_until_refresh"] = self._steps_until_refresh.unsqueeze(-1)
        return views

    def rgb_frame(self, world: int) -> torch.Tensor:
        """Return one world's cached uint8 HWC image."""
        return self._rgb[world]

    def depth_frame(self, world: int) -> torch.Tensor:
        """Return one world's cached camera-axis depth in meters."""
        return self._depth[world]

    def read(self, state, fields):
        values = {}
        if "rgb" in fields:
            values["rgb"] = self._rgb.clone()
        if "depth" in fields:
            values["depth"] = self._depth.clone()
        if "valid" in fields:
            values["valid"] = self._depth > 0
        return TensorDict(values, batch_size=[self.model.world_count])
