"""NewRL playback controls layered on Newton's Viser viewer."""

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from newrl.sensors import DepthCamera, GaussianCamera

if TYPE_CHECKING:
    import newton
    from newton.viewer import ViewerViser
    from newrl.scene import Scene
    from viser import ClientHandle
    import viser

_FOLLOW_INTERVAL = 0.06
_ORBIT_SETTLE_TIME = 0.2


@dataclass
class _ClientView:
    world: "viser.GuiDropdownHandle"
    robot: "viser.GuiDropdownHandle"
    follow: "viser.GuiCheckboxHandle"
    show_frustums: "viser.GuiCheckboxHandle | None"
    camera: "viser.GuiDropdownHandle | None" = None
    show_gaussians: "viser.GuiCheckboxHandle | None" = None
    gaussians: dict[str, "viser.GaussianSplatHandle"] = field(default_factory=dict)
    frustums: dict[str, "viser.CameraFrustumHandle"] = field(default_factory=dict)
    follow_target: tuple[int, str] | None = None
    orbit: np.ndarray | None = None
    follow_after: float = 0.0

    def follow_robot(
        self,
        camera: "viser.CameraHandle",
        selection: tuple[int, str],
        target: np.ndarray,
        now: float,
    ) -> None:
        orbit = camera.position - camera.look_at
        if self.follow_target == selection:
            # Reports from an orbit or zoom change the eye-to-target vector.
            # Our own follow updates translate both ends and preserve it.
            if not np.allclose(orbit, self.orbit, atol=1e-3, rtol=1e-3):
                self.orbit = orbit
                self.follow_after = now + _ORBIT_SETTLE_TIME
            if now < self.follow_after:
                return
        self.follow_target = selection
        self.orbit = orbit
        self.follow_after = now + _FOLLOW_INTERVAL
        # Viser translates look_at along with the eye.
        camera.position = target + orbit


class ViserPlayback:
    """Follow robots and draw sensor frustums and Gaussian clouds per client."""

    def __init__(self, viewer: "ViewerViser", scene: "Scene") -> None:
        self._server = viewer.server
        self._offsets = viewer.world_offsets.numpy()
        self._worlds = tuple(str(world) for world in range(scene.model.world_count))
        self._robots = scene.entities
        self._active_sensors = scene.physical_read_plan.keys()
        self._cameras = {
            robot: {
                name: sensor
                for name, sensor in scene.sensors.items()
                if isinstance(sensor, (DepthCamera, GaussianCamera)) and sensor.config.entity == robot
            }
            for robot in self._robots
        }
        self._select_camera = any(len(cameras) > 1 for cameras in self._cameras.values())
        self._clients: dict[int, _ClientView] = {}

    def update(self, state: "newton.State") -> None:
        clients = self._server.get_clients()
        for client_id in self._clients.keys() - clients.keys():
            del self._clients[client_id]
        for client_id, client in clients.items():
            view = self._clients.get(client_id)
            if view is None:
                with client.gui.add_folder("Robot view"):
                    view = _ClientView(
                        world=client.gui.add_dropdown("World", self._worlds),
                        robot=client.gui.add_dropdown("Robot", tuple(self._robots)),
                        follow=client.gui.add_checkbox("Follow robot", initial_value=False),
                        show_frustums=(
                            client.gui.add_checkbox("Camera frustums", initial_value=False)
                            if any(self._cameras.values())
                            else None
                        ),
                        camera=(
                            client.gui.add_dropdown("Camera", ("None",))
                            if self._select_camera else None
                        ),
                        show_gaussians=(
                            client.gui.add_checkbox("Gaussian scene", initial_value=False)
                            if any(
                                isinstance(sensor, GaussianCamera)
                                for cameras in self._cameras.values() for sensor in cameras.values()
                            ) else None
                        ),
                    )
                self._clients[client_id] = view

            world = int(view.world.value)
            robot_name = view.robot.value
            follow = view.follow.value
            if not follow:
                view.follow_target = None
            cameras = self._cameras[robot_name]
            if view.camera is not None:
                options = tuple(self._cameras[robot_name]) or ("None",)
                if view.camera.options != options:
                    view.camera.options = options
                cameras = {name: sensor for name, sensor in cameras.items() if name == view.camera.value}
            self._update_gaussians(client, view, cameras, world)
            if view.show_frustums is None or not view.show_frustums.value:
                cameras = {}
            for name in view.frustums.keys() - cameras.keys():
                view.frustums.pop(name).remove()
            if not follow and not cameras:
                continue

            body_q = wp.to_torch(state.body_q)
            poses = [
                body_q[sensor.body_ids[world:world + 1]][0] for sensor in cameras.values()
            ]
            if follow:
                robot = self._robots[robot_name]
                root = (
                    wp.to_torch(state.joint_q)[robot.root_q_indices[world]]
                    if robot.root_q_indices is not None
                    else body_q[robot.body_indices[world:world + 1, 0]][0]
                )
                poses.append(root)
            poses = torch.stack(poses).cpu().numpy()
            offset = self._offsets[world]

            with client.atomic():
                if follow:
                    target = poses[-1, :3] + offset
                    view.follow_robot(client.camera, (world, robot_name), target, monotonic())
                for (name, sensor), body_pose in zip(cameras.items(), poses):
                    transform = wp.transform(
                        wp.vec3(*body_pose[:3].tolist()), wp.quat(*body_pose[3:].tolist())
                    ) * sensor.mount_transform
                    position = wp.transform_get_translation(transform)
                    # Newton uses OpenGL optical axes. Viser uses OpenCV axes.
                    rotation = wp.transform_get_rotation(transform) * wp.quat(
                        1.0, 0.0, 0.0, 0.0
                    )
                    frustum = view.frustums.get(name)
                    if frustum is None:
                        config = sensor.config
                        frustum = client.scene.add_camera_frustum(
                            f"/newrl/cameras/{name}",
                            config.fov_y,
                            config.width / config.height,
                            color=(64, 192, 255),
                            format="png",
                        )
                        view.frustums[name] = frustum
                    frustum.position = tuple(
                        float(position[i]) + float(offset[i]) for i in range(3)
                    )
                    frustum.wxyz = (rotation[3], rotation[0], rotation[1], rotation[2])
                    if name in self._active_sensors:
                        if "rgb" in sensor.fields:
                            image = sensor.rgb_frame(world).cpu().numpy()
                        else:
                            depth = sensor.depth_frame(world).cpu().numpy().astype(np.float32)
                            far = sensor.config.far if isinstance(sensor, GaussianCamera) else sensor.config.max_depth
                            shade = 255 - 223 * np.clip(depth / far, 0, 1)
                            shade = np.where(depth > 0, shade, 0).astype(np.uint8)
                            image = np.repeat(shade[..., None], 3, axis=-1)
                        if not np.array_equal(frustum.image, image):
                            frustum.image = image

    def _update_gaussians(
        self,
        client: "ClientHandle",
        view: _ClientView,
        cameras: dict[str, DepthCamera | GaussianCamera],
        world: int,
    ) -> None:
        """Upload a cloud on demand and place it in the selected display world."""
        sensor = next(
            (sensor for sensor in cameras.values() if isinstance(sensor, GaussianCamera)),
            None,
        )
        enabled = view.show_gaussians is not None and view.show_gaussians.value
        source = str(Path(sensor.source).expanduser().resolve()) if enabled and sensor else None
        for path, handle in view.gaussians.items():
            handle.visible = path == source
        if source is None:
            return
        handle = view.gaussians.get(source)
        if handle is None:
            from gsrender.ply import read_ply
            from viser.transforms import SO3

            cloud = read_ply(Path(source))
            # Covariance = R diag(scale**2) R.T. Chunk conversion to bound
            # temporary memory for reconstructed scenes with millions of splats.
            covariances = np.empty((len(cloud.means), 3, 3), dtype=np.float32)
            for start in range(0, len(cloud.means), 65536):
                rows = slice(start, start + 65536)
                rotations = SO3(cloud.quats_xyzw[rows][:, [3, 0, 1, 2]]).as_matrix()
                axes = rotations * cloud.scales[rows, None, :]
                covariances[rows] = axes @ axes.transpose(0, 2, 1)
            handle = client.scene.add_gaussian_splats(
                f"/newrl/gaussians/{len(view.gaussians)}",
                centers=cloud.means,
                covariances=covariances,
                rgbs=np.clip(0.5 + 0.28209479177387814 * cloud.sh_dc, 0, 1),
                opacities=cloud.opacities[:, None],
            )
            view.gaussians[source] = handle
        transform = sensor.world_from_cloud
        handle.position = np.asarray(transform[:3]) + self._offsets[world]
        handle.wxyz = (transform[6], *transform[3:6])
