"""Copy selected cached camera images on the playback thread."""

from threading import Lock
from time import monotonic

import numpy as np

from newrl.sensors import DepthCamera


class CameraPreviews:
    def __init__(self, cameras, num_worlds):
        self.cameras = cameras
        self.num_worlds = num_worlds
        self._requested = {}
        self._frames = {}
        self._lock = Lock()
        self._next_sample = 0.0
        self._revision = 0

    def request(self, camera, world, field):
        sensor = self.cameras.get(camera)
        if sensor is None or not 0 <= world < self.num_worlds or field not in ("rgb", "depth"):
            raise KeyError
        if field not in sensor.fields:
            raise KeyError
        key = camera, world, field
        with self._lock:
            self._requested[key] = monotonic() + 0.75
            return self._frames.get(key)

    def sample(self):
        now = monotonic()
        if now < self._next_sample:
            return
        self._next_sample = now + 1 / 15
        with self._lock:
            for key, expiry in tuple(self._requested.items()):
                if expiry < now:
                    del self._requested[key]
                    self._frames.pop(key, None)
            requested = tuple(self._requested)
        for key in requested:
            camera, world, field = key
            sensor = self.cameras[camera]
            if field == "rgb":
                image = sensor.rgb_frame(world).cpu().numpy()
            else:
                depth = sensor.depth_frame(world).cpu().numpy().astype(np.float32)
                far = sensor.config.max_depth if isinstance(sensor, DepthCamera) else sensor.config.far
                image = np.where(depth > 0, 255 - 223 * np.clip(depth / far, 0, 1), 0).astype(np.uint8)
            payload = image.tobytes()
            with self._lock:
                previous = self._frames.get(key)
                if previous is None or previous[1] != payload:
                    self._revision += 1
                    self._frames[key] = f'"{self._revision}"', payload
