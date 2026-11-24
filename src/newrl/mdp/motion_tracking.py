"""Reference tracking with explicit clip progress and per-world reset RNG."""

from typing import Self, TYPE_CHECKING

from dataclasses import dataclass, field

import torch
import warp as wp
from tensordict import TensorDict

from newrl.datasets import LAFAN1Dataset
from newrl.utils.math import quat_rotate
from .base import MDP

if TYPE_CHECKING:
    from newrl.scene import Scene


@dataclass
class MotionTrackingMDP(MDP):
    entity: str = "robot"
    dataset: str | None = None
    root_body: str = "pelvis"
    goal_frames: int = 10
    seed: int = 0
    _motion: LAFAN1Dataset | None = field(default=None, init=False, repr=False)
    _root_com: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.goal_frames < 1:
            raise ValueError("goal_frames must be positive")

    def _initialize(self, scene: "Scene") -> Self:
        bound = super()._initialize(scene)
        robot = scene.entities[self.entity]
        if robot.root_q_indices is None or robot.root_qd_indices is None:
            raise ValueError("MotionTrackingMDP requires one free-root robot")
        device = robot.joint_q_indices.device
        motion = LAFAN1Dataset(self.dataset, fps=1 / scene.control_dt, device=device)
        joints = [name.rsplit("/", 1)[-1] for name in robot.joint_names]
        bodies = [name.rsplit("/", 1)[-1] for name in robot.body_names]
        if set(joints) != set(motion.joint_names):
            raise ValueError("Robot joints do not match the LAFAN1 G1 joint names")
        if set(bodies) - set(motion.body_names) or self.root_body not in bodies:
            raise ValueError("Robot bodies do not match the LAFAN1 motion skeleton")
        joint_ids = torch.tensor([motion.joint_names.index(name) for name in joints], device=device)
        body_ids = torch.tensor([motion.body_names.index(name) for name in bodies], device=device)
        for key in ("joint_pos", "joint_vel"):
            motion.data[key] = motion.data[key].index_select(-1, joint_ids)
        for key in ("body_pose_w", "body_twist_w"):
            motion.data[key] = motion.data[key].index_select(-2, body_ids)
        bound._motion = motion
        bound._root_com = wp.to_torch(scene.model.body_com)[robot.body_indices[0, bodies.index(self.root_body)]].clone()
        return bound

    def initial_state(self, num_envs: int, action_dim: int, device: torch.device) -> TensorDict:
        state = super().initial_state(num_envs, action_dim, device)
        worlds = torch.arange(num_envs, device=device, dtype=torch.long).unsqueeze(-1)
        state["reset_rng"] = ((self.seed % 2147483646) + worlds * 104729) % 2147483646 + 1
        state["motion_id"] = torch.zeros_like(worlds)
        state["motion_t"] = torch.zeros_like(worlds)
        state["clip_end"] = torch.zeros_like(worlds, dtype=torch.bool)
        return state

    @property
    def replay_fields(self) -> tuple[str, ...]:
        return (*super().replay_fields, "motion_id", "motion_t", "reset_rng")

    def restore_replay_state(self, state: TensorDict) -> TensorDict:
        restored = super().restore_replay_state(state)
        if self._motion is None:
            raise RuntimeError("Bind MotionTrackingMDP before restoring replay state")
        self._refresh_reference(restored)
        return restored

    def reset(
        self,
        mdp_state: TensorDict,
        ids: torch.Tensor | slice,
    ) -> tuple[TensorDict, dict[str, TensorDict]]:
        state, poses = super().reset(mdp_state, ids)
        if self._motion is None:
            raise RuntimeError("Bind MotionTrackingMDP before resetting")
        motion = self._motion
        rng = state["reset_rng"][ids]
        rng = (rng * 48271) % 2147483647
        motion_id = rng % motion.num_clips
        rng = (rng * 48271) % 2147483647
        length = motion.clip_end[motion_id] - motion.clip_start[motion_id]
        state["reset_rng"][ids] = rng
        state["motion_id"][ids] = motion_id
        state["motion_t"][ids] = rng % (length - 1)
        self._refresh_reference(state)
        reference = state["reference"][ids]
        twist = reference["root_twist"].clone()
        # Dataset velocities are link-origin referenced; Newton free DOFs use COMs.
        offset = quat_rotate(reference["root_pose"][..., 3:7], self._root_com.expand_as(twist[..., :3]))
        twist[..., :3] += torch.linalg.cross(twist[..., 3:], offset)
        poses[self.entity] = TensorDict({
            "root_pose": reference["root_pose"].clone(), "root_twist": twist,
            "joint_pos": reference["joint_pos"].clone(),
            "joint_vel": reference["joint_vel"].clone(),
        }, batch_size=reference.batch_size)
        return state, poses

    def _refresh_reference(self, state: TensorDict) -> None:
        motion = self._motion
        ids, frame = state["motion_id"].squeeze(-1), state["motion_t"].squeeze(-1)
        state["reference"] = motion.get_slice(ids, frame, torch.zeros(1, device=ids.device, dtype=torch.long))[..., 0]
        future = motion.get_slice(ids, frame, torch.arange(1, self.goal_frames + 1, device=ids.device), fields=("root_pose", "joint_pos"))
        state["future_reference"] = future.select("root_pose", "joint_pos")
        state["clip_end"] = state["motion_t"] >= (motion.clip_end[ids] - motion.clip_start[ids] - 1).unsqueeze(-1)

    def _advance_task_state(
        self,
        physical_state: TensorDict,
        state: TensorDict,
    ) -> dict[str, torch.Tensor]:
        ids = state["motion_id"].squeeze(-1)
        last = (self._motion.clip_end[ids] - self._motion.clip_start[ids] - 1).unsqueeze(-1)
        state["motion_t"] = torch.minimum(state["motion_t"] + 1, last)
        self._refresh_reference(state)
        return {}
