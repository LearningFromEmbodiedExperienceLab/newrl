"""Official G1-retargeted LAFAN1 CSVs, resampled and cached for tracking.

LAFAN1_DATASET selects a CSV directory (or one CSV), or a processed NPZ.
Otherwise the pinned official G1 CSVs are fetched into <repo>/.cache/lafan1.
All processed quaternions are xyzw; twists are world-frame, link-origin
linear velocity followed by angular velocity. Source CSVs are 30 Hz.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
JOINT_NAMES = tuple(
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
)
FIELDS = ("root_pose", "root_twist", "joint_pos", "joint_vel", "body_pose_w", "body_twist_w")


def _download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete files; a failed download never becomes a cache hit.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
            output.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _angular_velocity(quat: np.ndarray, fps: float) -> np.ndarray:
    """World angular velocities from shortest quaternion increments (xyzw)."""
    first, second = quat[:-1], quat[1:]
    vector = (
        first[..., 3:] * second[..., :3] - second[..., 3:] * first[..., :3]
        - np.cross(second[..., :3], first[..., :3])
    )
    scalar = (first * second).sum(-1, keepdims=True)
    vector = np.where(scalar < 0, -vector, vector)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, np.abs(scalar))
    velocity = vector * (angle / np.maximum(norm, 1e-10)) * fps
    # Match np.gradient's centered interior / one-sided endpoint convention.
    return np.concatenate((velocity[:1], (velocity[:-1] + velocity[1:]) / 2, velocity[-1:]), axis=0)


class LAFAN1Dataset:
    def __init__(self, path=None, *, model_path=None, fps=50.0, device="cpu", cache_dir=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Motion fps must be finite and positive")
        cache = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[3] / ".cache" / "lafan1"
        source = path if path is not None else os.environ.get("LAFAN1_DATASET")
        if source is None:
            source = cache / REVISION / "g1"
            source.mkdir(parents=True, exist_ok=True)
            with (source.parent / ".download.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                manifest = source.parent / "manifest.json"
                if not manifest.exists():
                    _download(f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}", manifest)
                files = json.loads(manifest.read_text())["siblings"]
                names = sorted(item["rfilename"] for item in files if item["rfilename"].startswith("g1/") and item["rfilename"].endswith(".csv"))
                if not names:
                    raise ValueError("Official LAFAN1 revision contains no G1 CSVs")
                for name in ["README.md", "LICENSE", *names]:
                    destination = source.parent / name
                    if not destination.exists():
                        print(f"Downloading LAFAN1: {name}", flush=True)
                        _download(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}", destination)
        source = Path(source).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"LAFAN1_DATASET does not exist: {source}")
        if source.suffix == ".npz":
            processed = source
        else:
            files = [source] if source.is_file() else sorted(source.glob("*.csv"))
            if not files or any(file.suffix != ".csv" for file in files):
                raise ValueError("LAFAN1_DATASET must be a CSV, CSV directory, or processed NPZ")
            if model_path is None:
                from newrl.assets import cooked_robot
                from newrl.robots import robot_model_dir

                model_path = cooked_robot("g1", robot_model_dir()) / "model.xml"
            digest = hashlib.sha256(Path(model_path).read_bytes())
            digest.update(f"lafan1-v1:{fps}".encode())
            for file in files:
                digest.update(file.name.encode())
                digest.update(file.read_bytes())
            processed = cache / f"g1-{digest.hexdigest()[:20]}.npz"
            cache.mkdir(parents=True, exist_ok=True)
            with processed.with_suffix(".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not processed.exists():
                    self._process(files, Path(model_path), fps, processed)
        with np.load(processed, allow_pickle=False) as data:
            if int(data["version"]) != 1 or not np.isclose(float(data["fps"]), fps):
                raise ValueError("Processed LAFAN1 format or frame rate does not match the task")
            self.fps = float(data["fps"])
            self.joint_names = tuple(data["joint_names"].tolist())
            self.body_names = tuple(data["body_names"].tolist())
            self.clip_names = tuple(data["clip_names"].tolist())
            starts, ends = data["starts"], data["ends"]
            count = len(data["root_pose"])
            if (not len(starts) or starts.shape != ends.shape or len(starts) != len(self.clip_names)
                or starts.dtype.kind not in "iu" or ends.dtype.kind not in "iu"
                or starts[0] != 0 or ends[-1] != count or np.any(ends - starts < 2)
                or not np.array_equal(starts[1:], ends[:-1])):
                raise ValueError("Invalid LAFAN1 clip ranges")
            for names in (self.joint_names, self.body_names, self.clip_names):
                if not names or len(set(names)) != len(names):
                    raise ValueError("Motion names must be nonempty and unique")
            shapes = ((7,), (6,), (len(self.joint_names),), (len(self.joint_names),), (len(self.body_names), 7), (len(self.body_names), 6))
            tensors = {}
            for name, shape in zip(FIELDS, shapes, strict=True):
                values = data[name]
                if values.shape != (count, *shape) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid motion field {name}")
                if "pose" in name and not np.allclose(np.linalg.norm(values[..., 3:7], axis=-1), 1, atol=1e-4):
                    raise ValueError(f"Motion field {name} requires unit xyzw quaternions")
                tensors[name] = torch.tensor(values, dtype=torch.float32, device=device)
            self.data = TensorDict(tensors, batch_size=[count], device=device)
            self.clip_start = torch.tensor(starts, dtype=torch.long, device=device)
            self.clip_end = torch.tensor(ends, dtype=torch.long, device=device)
        self.num_clips = len(self.clip_names)
        self.path = processed

    @staticmethod
    def _process(files, model_path, fps, destination):
        model = mujoco.MjModel.from_xml_path(str(model_path))
        joints = [model.joint(name) for name in JOINT_NAMES]
        if model.nq != 36 or model.nv != 35:
            raise ValueError("LAFAN1 G1 processing requires the 29-DoF free-root model")
        body_ids = np.arange(1, model.nbody)
        body_names = [model.body(int(body)).name for body in body_ids]
        clips = []
        for file in files:
            print(f"Processing LAFAN1: {file.name}", flush=True)
            raw = np.loadtxt(file, delimiter=",")
            if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
                raise ValueError(f"Expected finite [frames,36] root XY Z/xyzw and G1 joint CSV: {file}")
            quat = raw[:, 3:7]
            lengths = np.linalg.norm(quat, axis=-1, keepdims=True)
            if np.any(lengths < 1e-8):
                raise ValueError(f"Invalid zero root quaternion in {file}")
            quat = quat / lengths
            # Continuous quaternion signs followed by shortest-arc SLERP.
            signs = np.where((quat[1:] * quat[:-1]).sum(-1) < 0, -1, 1).cumprod()
            quat[1:] *= signs[:, None]
            time = np.arange(int(np.floor((len(raw) - 1) * fps / 30)) + 1) / fps
            if len(time) < 2:
                raise ValueError(f"Motion {file} is too short for the requested frame rate")
            source_time = np.arange(len(raw)) / 30
            coordinates = np.stack([np.interp(time, source_time, raw[:, column]) for column in range(36)], -1)
            index = np.minimum((time * 30).astype(int), len(raw) - 2)
            fraction = (time * 30 - index)[:, None]
            a, b = quat[index], quat[index + 1]
            angle = np.arccos((a * b).sum(-1, keepdims=True).clip(-1, 1))
            sine = np.sin(angle)
            rotations = (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.maximum(sine, 1e-8)
            rotations = np.where(angle < 1e-6, (1 - fraction) * a + fraction * b, rotations)
            coordinates[:, 3:7] = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)
            state = mujoco.MjData(model)
            poses = np.empty((len(time), len(body_ids), 7))
            for frame, values in enumerate(coordinates):
                state.qpos[:3] = values[:3]
                state.qpos[3:7] = values[[6, 3, 4, 5]]
                for column, joint in enumerate(joints):
                    state.qpos[joint.qposadr[0]] = values[7 + column]
                mujoco.mj_kinematics(model, state)
                poses[frame, :, :3] = state.xpos[body_ids]
                poses[frame, :, 3:] = state.xquat[body_ids][:, [1, 2, 3, 0]]
            root_pose = coordinates[:, :7]
            root_twist = np.concatenate((np.gradient(root_pose[:, :3], 1 / fps, axis=0), _angular_velocity(root_pose[:, 3:], fps)), -1)
            body_twist = np.concatenate((np.gradient(poses[..., :3], 1 / fps, axis=0), _angular_velocity(poses[..., 3:], fps)), -1)
            clips.append(dict(zip(FIELDS, (root_pose, root_twist, coordinates[:, 7:], np.gradient(coordinates[:, 7:], 1 / fps, axis=0), poses, body_twist), strict=True)))
        ends = np.cumsum([len(clip["root_pose"]) for clip in clips])
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as output:
            temporary = Path(output.name)
        try:
            np.savez(temporary, version=1, fps=fps, joint_names=JOINT_NAMES, body_names=body_names,
                     clip_names=[file.stem for file in files], starts=np.r_[0, ends[:-1]], ends=ends,
                     **{name: np.concatenate([clip[name] for clip in clips]).astype(np.float32) for name in FIELDS})
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def get_slice(self, motion_id, motion_t, offsets, fields=FIELDS):
        """Return TD[*batch,T], clamped independently inside each selected clip.

        motion_id and motion_t are integer [*batch] indices (strip scalar
        column dimensions from MDP state before calling).
        """
        start = self.clip_start[motion_id]
        index = start[..., None] + motion_t[..., None] + offsets
        index = torch.maximum(start[..., None], torch.minimum(index, self.clip_end[motion_id][..., None] - 1))
        return self.data.select(*fields)[index]
