"""Local Three.js playback for Newton meshes and static Gaussian scenes."""

import base64
import gzip
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import SubprocessError, run
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

import newton
import numpy as np

from newrl.sensors import DepthCamera, GaussianCamera
from newrl.utils.gaussian_scene import resolve_gaussian_scene
from .camera_preview import CameraPreviews
from .markers import Marker, Pose, RobotGhost, check_pose

if TYPE_CHECKING:
    from torch import Tensor


def _pack(values, dtype):
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype)).decode("ascii")


class ViewportRecordings:
    """Keep recordings private to this viewer until the client finalizes them."""

    def __init__(self, directory, task_name):
        self.directory = Path(directory).expanduser()
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_name).strip("_")
        self.task_name = safe_name[:80] or "viewport"
        self.active = {}
        self.lock = Lock()
        self.closed = False

    def start(self, extension):
        with self.lock:
            if self.closed:
                raise ValueError("Viewer is shutting down")
            self.directory.mkdir(parents=True, exist_ok=True)
            recording_id = uuid4().hex
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final = self.directory / f"{self.task_name}_{timestamp}_{recording_id[:12]}.{extension}"
            partial = final.with_suffix(final.suffix + ".part")
            partial.touch(exist_ok=False)
            self.active[recording_id] = (partial, final)
            return {"id": recording_id}

    def update(self, recording_id, action, chunk):
        with self.lock:
            if action == "cancel":
                paths = self.active.pop(recording_id, None)
                if paths is not None:
                    paths[0].unlink(missing_ok=True)
                return {}
            partial, final = self.active[recording_id]
            if action == "chunk":
                with partial.open("ab") as stream:
                    stream.write(chunk)
                return {}
            if not partial.stat().st_size:
                raise ValueError("The browser captured no video frames")
            partial.rename(final)
            del self.active[recording_id]
            return {"path": str(final)}

    def close(self):
        with self.lock:
            self.closed = True
            for partial, _ in self.active.values():
                partial.unlink(missing_ok=True)
            self.active.clear()


class BrowserViewer(newton.viewer.ViewerNull):
    """Publish the latest pose snapshot and serve local viewer assets."""

    def __init__(
        self, scene, port, splat_path=None, *, recording_dir="recordings",
        task_name="viewport", timeline=None, terrain_origin_frames=True,
    ):
        assets = Path(__file__).with_name("browser_assets")
        if not (assets / "index.html").is_file():
            raise RuntimeError(
                "Build the browser viewer with npm --prefix viewer ci and npm --prefix viewer run build"
            )
        protocol = assets / "protocol.txt"
        if not protocol.is_file() or protocol.read_text().strip() != "3":
            raise RuntimeError(
                "Browser viewer assets are out of date. "
                "Run npm --prefix viewer run build in the NewRL checkout."
            )
        super().__init__()
        self.meshes = {}
        self.instances = {}
        self.poses = {}
        self.manifest = None
        self.manifest_gzip = None
        self.frame = None
        self.frame_id = 0
        self.frame_epoch = 0
        self.started_at = None
        self._last_frame_time = None
        self._paused = Event()
        self.running = True
        self.timeline = timeline
        self.num_worlds = scene.model.world_count
        self.recordings = ViewportRecordings(recording_dir, task_name)
        self.clouds = []
        self.cloud_paths = {}
        self._markers = {}
        self._ghosts = {}
        self._ghost_poses = {}
        self._marker_poses = {}
        self._marker_vectors = {}
        self._marker_points = {}
        self._marker_bodies = {}
        self._body_world = scene.model.body_world.numpy()
        self.camera_previews = CameraPreviews(scene.cameras, scene.model.world_count)
        for i, (x, y, z) in enumerate(scene.terrain.tile_origins):
            self.add_marker(Marker(
                name=f"Tile {i + 1}", kind="frame", pose=(x, y, z, 0, 0, 0, 1),
                size=0.4, group="Terrain origins", visible=terrain_origin_frames,
            ))
        self.gpu_stats = b"null"
        self._stop_stats = Event()
        self._terrain_shapes = set(scene.terrain_shapes)
        self._body_names = {
            body: name
            for name, entity in scene.entities.items()
            for body in entity.body_indices.flatten().tolist()
        }
        self._shape_bodies = scene.model.shape_body.numpy()
        self.targets = [
            (world, name, int(entity.body_indices[world, 0].item()))
            for world in range(scene.model.world_count)
            for name, entity in scene.entities.items()
        ]
        sources = []
        self.cameras = []
        camera_bodies = []
        if splat_path is not None:
            source, pose = resolve_gaussian_scene(splat_path, browser=True)
            sources.append(("Gaussian scene", source, pose))
        for name, sensor in scene.sensors.items():
            if isinstance(sensor, (DepthCamera, GaussianCamera)):
                config = sensor.config
                self.cameras.append({
                    "name": name,
                    "mount": list(sensor.mount_transform),
                    "fovY": config.fov_y,
                    "aspect": config.width / config.height,
                    "width": config.width,
                    "height": config.height,
                    "previewFields": [
                        field for field in sensor.fields if field in ("rgb", "depth")
                    ] if name in scene.cameras else [],
                })
                camera_bodies.append(sensor.body_ids.tolist())
            if isinstance(sensor, GaussianCamera):
                sources.append(
                    (name, sensor.source, sensor.world_from_cloud)
                )
        self._camera_bodies = np.asarray(camera_bodies, dtype=np.intp).reshape(
            len(self.cameras), scene.model.world_count
        )
        for name, source, transform in sources:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Gaussian scene file does not exist: {path}")
            route = f"/cloud/{len(self.clouds)}{path.suffix.lower()}"
            self.cloud_paths[route] = path
            self.clouds.append(
                {"name": name, "url": route, "transform": list(transform)}
            )

        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                port = self.server.server_port
                if self.headers.get("Origin") not in (
                    f"http://127.0.0.1:{port}", f"http://localhost:{port}",
                ):
                    self.send_error(403)
                    return
                parts = urlsplit(self.path).path.strip("/").split("/")
                result = {}
                status = 200
                try:
                    if parts in (["playback", "pause"], ["playback", "resume"]):
                        viewer.set_paused(parts[1] == "pause")
                        result = {"paused": viewer.is_paused()}
                    elif timeline is not None and len(parts) == 3 and parts[:2] == ["timeline", "branch"]:
                        timeline.request_branch(int(parts[2]))
                    else:
                        result = self._recording_request(parts)
                except KeyError:
                    status, result = 404, {"error": "Recording or timeline frame not found"}
                except (ValueError, TimeoutError) as error:
                    status, result = 400, {"error": str(error)}
                except OSError as error:
                    status, result = 500, {"error": str(error)}
                payload = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _recording_request(self, parts):
                if len(parts) != 3 or parts[0] != "recordings":
                    raise KeyError
                _, recording_id, action = parts
                if recording_id == "start" and action in ("mp4", "webm"):
                    return viewer.recordings.start(action)
                if action not in ("chunk", "finish", "cancel"):
                    raise KeyError
                chunk = None
                if action == "chunk":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4 * 1024 * 1024:
                        raise ValueError("Invalid video chunk size")
                    self.connection.settimeout(10)
                    chunk = self.rfile.read(size)
                    if len(chunk) != size:
                        raise ValueError("Incomplete video chunk")
                return viewer.recordings.update(recording_id, action, chunk)

            def do_GET(self):
                route = urlsplit(self.path).path
                if route.startswith("/camera/"):
                    try:
                        _, camera, world, field = route.lstrip("/").split("/")
                        camera = int(camera)
                        if not 0 <= camera < len(viewer.cameras):
                            raise ValueError
                        name = viewer.cameras[camera]["name"]
                        cached = viewer.camera_previews.request(name, int(world), field)
                    except (KeyError, ValueError):
                        self.send_error(404, "Camera preview unavailable")
                        return
                    if cached is None:
                        self.send_response(204)
                        self.end_headers()
                    elif self.headers.get("If-None-Match") == cached[0]:
                        self.send_response(304)
                        self.end_headers()
                    else:
                        tag, payload = cached
                        self.send_response(200)
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(len(payload)))
                        self.send_header("ETag", tag)
                        self.end_headers()
                        self.wfile.write(payload)
                    return
                if timeline is not None and route == "/timeline.json":
                    self._send_json(json.dumps(timeline.index()).encode())
                    return
                if timeline is not None and route.startswith("/timeline/frame/"):
                    try:
                        frame = timeline.frame(int(route.removeprefix("/timeline/frame/")))
                    except (ValueError, KeyError):
                        self.send_error(404, "Timeline frame is no longer available")
                        return
                    self._send_json(frame.poses, headers={
                        "X-Timeline-Branchable": str(int(frame.checkpoint is not None)),
                    })
                    return
                if route in ("/scene.json", "/frame.json", "/gpu.json"):
                    payload = {
                        "/scene.json": viewer.manifest,
                        "/frame.json": viewer.frame,
                        "/gpu.json": viewer.gpu_stats,
                    }[route]
                    headers = {"X-Simulation-Paused": str(int(viewer.is_paused()))} if route == "/frame.json" else {}
                    if route == "/scene.json" and payload is not None and self._accepts_gzip():
                        payload = viewer.manifest_gzip
                        headers["Content-Encoding"] = "gzip"
                    self._send_json(payload, headers=headers)
                    return
                path = viewer.cloud_paths.get(route)
                if route == "/":
                    path = assets / "index.html"
                elif route.startswith("/assets/"):
                    candidate = (assets / route.lstrip("/")).resolve()
                    if candidate.is_relative_to(assets.resolve()):
                        path = candidate
                if path is None or not path.is_file():
                    self.send_error(404)
                    return
                types = {
                    ".html": "text/html",
                    ".css": "text/css",
                    ".js": "text/javascript",
                    ".wasm": "application/wasm",
                }
                compressed = path.with_name(path.name + ".gz")
                use_gzip = route.startswith("/assets/") and compressed.is_file() and self._accepts_gzip()
                source = compressed if use_gzip else path
                with source.open("rb") as stream:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        types.get(path.suffix, "application/octet-stream"),
                    )
                    self.send_header("Content-Length", str(source.stat().st_size))
                    self.send_header("Vary", "Accept-Encoding")
                    if use_gzip:
                        self.send_header("Content-Encoding", "gzip")
                    if route == "/":
                        self.send_header("Cache-Control", "no-cache")
                    elif re.fullmatch(r"/assets/[^/]+-[\w-]+\.(js|css)", route):
                        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                    self.end_headers()
                    shutil.copyfileobj(stream, self.wfile)

            def _accepts_gzip(self):
                for item in self.headers.get("Accept-Encoding", "").split(","):
                    encoding, *parameters = item.strip().lower().split(";")
                    if encoding == "gzip":
                        try:
                            return all(float(p.split("=", 1)[1]) > 0 for p in parameters if p.strip().startswith("q="))
                        except ValueError:
                            return False
                return False

            def _send_json(self, payload, *, headers=()):
                self.send_response(200 if payload is not None else 204)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Vary", "Accept-Encoding")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload or b"")))
                for name, value in dict(headers).items():
                    self.send_header(name, value)
                self.end_headers()
                if payload is not None:
                    self.wfile.write(payload)

            def log_message(self, *args):
                pass

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            if port == 0:
                raise
            print(
                f"Viewer port {port} is unavailable. Requesting an available port.",
                flush=True,
            )
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._stats_thread = None
        if scene.model.device.uuid and (smi := shutil.which("nvidia-smi")):
            self._stats_thread = Thread(
                target=self._sample_gpu, args=(smi, scene.model.device.uuid), daemon=True
            )
            self._stats_thread.start()
        print(
            f"NewRL browser viewer running at: http://127.0.0.1:{self.server.server_port}",
            flush=True,
        )

    def _sample_gpu(self, smi, device_uuid):
        command = [
            smi, "-i", device_uuid,
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_stats.is_set():
            try:
                sample = run(command, check=True, capture_output=True, text=True, timeout=1)
                utilization, used, total = (
                    int(value.strip()) if value.strip().isdigit() else None
                    for value in sample.stdout.strip().split(",")
                )
                self.gpu_stats = json.dumps({
                    "utilization": utilization, "usedMiB": used, "totalMiB": total,
                }).encode()
            except (OSError, SubprocessError, ValueError):
                self.gpu_stats = b"null"
            self._stop_stats.wait(1)

    def set_model(self, model):
        super().set_model(model)
        self._batches_by_name = {batch.name: batch for batch in self._shape_instances.values()}

    def add_marker(self, marker: Marker) -> None:
        """Register a marker before publishing the first viewport frame."""
        if self.manifest is not None or marker.name in self._markers or marker.name in self._ghosts:
            raise ValueError("Register each marker once before the first frame")
        if marker.body is not None and not 0 <= marker.body < len(self._body_world):
            raise ValueError("Marker body index is out of range")
        self._markers[marker.name] = marker

    def add_ghost(self, ghost: RobotGhost, body_ids: list[int]) -> None:
        """Reuse world zero's visual meshes as a template for each world's reference."""
        if self.manifest is not None or ghost.name in self._ghosts or ghost.name in self._markers:
            raise ValueError("Register each marker once before the first frame")
        self._ghosts[ghost.name] = ghost, dict(zip(body_ids, range(len(body_ids)), strict=True))

    def set_ghost_poses(self, name: str, poses: "Tensor") -> None:
        _, slots = self._ghosts[name]
        if poses.shape != (self.num_worlds, len(slots), 7):
            raise ValueError("Ghost poses require shape [num_worlds, num_bodies, 7]")
        values = poses.detach().cpu().numpy()
        self._ghost_poses[name] = _pack(values, "<f4") if np.isfinite(values).all() else None

    def _ghost_manifest(self):
        specs = []
        for ghost, slots in self._ghosts.values():
            shapes = []
            for batch in self._batches_by_name.values():
                if batch.mesh not in self.meshes or not batch.flags & int(newton.ShapeFlags.VISIBLE):
                    continue
                matching = [
                    (i, slots[parent])
                    for i, shape in enumerate(batch.model_shapes)
                    if (parent := self._shape_bodies[shape]) in slots
                ]
                if not matching:
                    continue
                indices, bodies = zip(*matching, strict=True)
                indices = list(indices)
                poses = batch.xforms.numpy()[indices].tolist()
                scales = batch.scales.numpy()[indices].tolist()
                shapes.extend(
                    {"mesh": batch.mesh, "body": body, "pose": pose, "scale": scale}
                    for body, pose, scale in zip(bodies, poses, scales, strict=True)
                )
            specs.append({**asdict(ghost), "bodyCount": len(slots), "shapes": shapes})
        return specs

    def set_marker_pose(self, name: str, pose: Pose) -> None:
        """Update a marker in its original coordinate space for subsequent frames."""
        if name not in self._markers:
            raise KeyError(name)
        check_pose(pose)
        self._marker_poses[name] = pose

    def set_marker_vectors(self, name: str, vectors: "Tensor") -> None:
        """Supply XYZ origin and vector per world to a world-space arrow."""
        marker = self._markers[name]
        if marker.kind != "arrow" or marker.body is not None:
            raise ValueError("Batched vectors require a world-space arrow marker")
        if vectors.shape != (self.num_worlds, 6):
            raise ValueError("Marker vectors require shape [num_worlds, 6]")
        values = vectors.detach().cpu().numpy()
        self._marker_vectors[name] = np.where(
            np.isfinite(values).all(axis=1, keepdims=True), values, 0
        ).tolist()

    def set_marker_points(self, name: str, positions: "Tensor", valid: "Tensor") -> None:
        """Supply world-space point positions and a validity mask for each world."""
        marker = self._markers[name]
        if marker.kind != "points":
            raise ValueError("Batched positions require a points marker")
        if positions.ndim != 3 or positions.shape[0] != self.num_worlds or positions.shape[-1] != 3:
            raise ValueError("Marker points require shape [num_worlds, num_points, 3]")
        if valid.shape != positions.shape[:2]:
            raise ValueError("Marker validity requires shape [num_worlds, num_points]")
        values = positions.detach().cpu().numpy()
        mask = valid.detach().cpu().numpy().astype(bool) & np.isfinite(values).all(axis=-1)
        self._marker_points[name] = [
            points[keep].tolist() for points, keep in zip(values, mask, strict=True)
        ]

    def _should_show_shape(self, flags, is_static, geo_type=None, *, layer=None):
        # Publish collision poses for client-side visibility without enabling
        # Newton's extra collision debugging geometry.
        return bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) or super()._should_show_shape(
            flags, is_static, geo_type, layer=layer
        )

    def _instance_layers(self, name, count):
        batch = self._batches_by_name.get(name)
        if batch is None:
            return [["Scene geometry"]] * count
        layers = []
        for shape in batch.model_shapes:
            if shape in self._terrain_shapes:
                layers.append(["Ground"])
                continue
            entity = self._body_names.get(self._shape_bodies[shape], "Scene geometry")
            categories = []
            if batch.flags & int(newton.ShapeFlags.VISIBLE):
                categories.append(f"{entity} visuals")
            if batch.flags & int(newton.ShapeFlags.COLLIDE_SHAPES):
                categories.append(f"{entity} collision")
            layers.append(categories)
        return layers

    def log_mesh(self, name, points, indices, normals=None, uvs=None, **kwargs):
        self.meshes[name] = {
            "points": _pack(points.numpy(), "<f4"),
            "indices": _pack(indices.numpy(), "<u4"),
            "normals": _pack(normals.numpy(), "<f4") if normals is not None else None,
        }

    def log_instances(
        self,
        name,
        mesh,
        xforms,
        scales,
        colors,
        materials,
        hidden=False,
        opacities=None,
    ):
        if hidden or xforms is None:
            return
        self.poses[name] = _pack(xforms.numpy(), "<f4")
        if self.manifest is None:
            self.instances[name] = {
                "mesh": mesh,
                "count": len(xforms),
                "layers": self._instance_layers(name, len(xforms)),
                "scales": scales.numpy().tolist() if scales is not None else None,
                "colors": colors.numpy().tolist() if colors is not None else None,
                "opacities": opacities.numpy().tolist()
                if opacities is not None
                else None,
            }

    def log_state(self, state):
        if self.time == self._last_frame_time:
            return
        super().log_state(state)
        body_poses = state.body_q.numpy()
        self.body_positions = body_poses[[body for _, _, body in self.targets], :3]
        self.camera_body_poses = body_poses[self._camera_bodies]
        self._marker_bodies = {
            marker.body: body_poses[marker.body].tolist()
            for marker in self._markers.values()
            if marker.body is not None
        }

    def begin_frame(self, time, *, jump=False):
        self.time = time
        if jump:
            self.frame_epoch += 1
            self._last_frame_time = None
        if self.started_at is None or jump:
            self.started_at = monotonic() - time

    def end_frame(self, *, pace=True):
        self.camera_previews.sample()
        if self.time == self._last_frame_time:
            sleep(0.02)
            # Keep playback paced after a pause instead of catching up to wall time.
            self.started_at = monotonic() - self.time
            return
        self._last_frame_time = self.time
        offsets = self.world_offsets.numpy() if self.manifest is None else self._offsets
        self._offsets = offsets
        if self.manifest is None:
            manifest = json.dumps(
                {
                    "protocol": 3,
                    "meshes": self.meshes,
                    "instances": self.instances,
                    "clouds": self.clouds,
                    "cameras": self.cameras,
                    "ghosts": self._ghost_manifest(),
                    "markers": [
                        {
                            **asdict(marker),
                            "offset": offsets[self._body_world[marker.body]].tolist()
                            if marker.body is not None and self._body_world[marker.body] >= 0
                            else [0, 0, 0],
                        }
                        for marker in self._markers.values()
                    ],
                    "timeline": self.timeline is not None,
                    "targets": [
                        {
                            "world": world,
                            "name": name,
                            "offset": offsets[world].tolist(),
                        }
                        for world, name, _ in self.targets
                    ],
                },
                separators=(",", ":"),
            ).encode()
            self.manifest_gzip = gzip.compress(manifest, compresslevel=1)
            self.manifest = manifest
        targets = [
            position + offsets[world]
            for position, (world, _, _) in zip(
                self.body_positions, self.targets, strict=True
            )
        ]
        self.frame_id += 1
        self.frame = json.dumps(
            {
                "id": self.frame_id,
                "epoch": self.frame_epoch,
                "time": self.time,
                "poses": self.poses,
                "targets": [position.tolist() for position in targets],
                "cameraBodies": self.camera_body_poses.tolist(),
                "markerPoses": self._marker_poses,
                "markerVectors": self._marker_vectors,
                "markerPoints": self._marker_points,
                "markerBodies": self._marker_bodies,
                "ghostPoses": self._ghost_poses,
            },
            separators=(",", ":"),
        ).encode()
        remaining = self.started_at + self.time - monotonic()
        if pace and remaining > 0:
            sleep(remaining)

    def is_running(self):
        return self.running

    def is_paused(self):
        return self._paused.is_set()

    def set_paused(self, paused):
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def close(self):
        self.running = False
        self._stop_stats.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        if self._stats_thread is not None:
            self._stats_thread.join()
        self.recordings.close()
