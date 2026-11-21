"""Velocity tracking with task-owned command and physical reset sampling."""

from dataclasses import dataclass, field

import torch
import warp as wp
from tensordict import TensorDict

from .base import MDP


@dataclass
class VelocityMDP(MDP):
    entity: str = "robot"
    vx: tuple[float, float] = (-0.5, 1.0)
    vy: tuple[float, float] = (-0.5, 0.5)
    yaw_rate: tuple[float, float] = (-1.0, 1.0)
    hold_steps: int = 250
    seed: int = 0
    joint_position_noise: float = 0.05
    root_xy_noise: float = 0.10
    root_yaw_noise: float = 0.15
    _nominal_pose: TensorDict | None = field(default=None, init=False, repr=False)
    _spawn_locations: torch.Tensor | None = field(default=None, init=False, repr=False)
    _joint_limits: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.hold_steps < 1:
            raise ValueError("hold_steps must be positive")
        if any(lo > hi for lo, hi in (self.vx, self.vy, self.yaw_rate)):
            raise ValueError("Command ranges must be ordered")
        if min(self.joint_position_noise, self.root_xy_noise, self.root_yaw_noise) < 0:
            raise ValueError("Reset noise amplitudes must be nonnegative")

    def _initialize(self, scene):
        bound = super()._initialize(scene)
        robot = scene.entities[self.entity]
        if robot.root_q is None:
            raise ValueError("VelocityMDP requires a robot with one free root joint")
        bound._nominal_pose = TensorDict(
            {
                "root_pose": wp.to_torch(scene.model.joint_q)[robot.root_q],
                "joint_pos": robot.default_position,
            },
            batch_size=[len(robot.q)],
            device=robot.q.device,
        )
        bound._spawn_locations = scene.spawn_locations
        bound._joint_limits = (
            wp.to_torch(scene.model.joint_limit_lower)[robot.qd],
            wp.to_torch(scene.model.joint_limit_upper)[robot.qd],
        )
        return bound

    def initial_state(self, num_envs, action_dim, device):
        mdp_state = super().initial_state(num_envs, action_dim, device)
        worlds = torch.arange(num_envs, device=device, dtype=torch.long).unsqueeze(-1)
        mdp_state["command_rng"] = (
            (self.seed % 2147483646) + worlds * 104729
        ) % 2147483646 + 1
        # Separate streams keep physical pose samples independent of the number
        # of command changes during the previous episode.
        mdp_state["reset_rng"] = (
            mdp_state["command_rng"] * 48271 + 7919
        ) % 2147483646 + 1
        mdp_state["command_steps_left"] = torch.zeros_like(worlds)
        mdp_state["velocity_command"] = mdp_state["previous_action"].new_zeros(
            (num_envs, 3)
        )
        mdp_state["spawn_height"] = mdp_state["velocity_command"].new_zeros(
            (num_envs, 1)
        )
        return mdp_state

    def reset(self, mdp_state, ids):
        next_mdp_state, poses = super().reset(mdp_state, ids)
        if self._nominal_pose is None or self._joint_limits is None:
            raise RuntimeError(
                "Bind VelocityMDP to a scene before sampling reset poses"
            )
        selected = torch.zeros_like(
            next_mdp_state["command_steps_left"], dtype=torch.bool
        )
        selected[ids] = True
        self._sample_command(next_mdp_state, selected)

        pose = self._nominal_pose[ids].clone()
        root, joints = pose["root_pose"], pose["joint_pos"]
        rng = next_mdp_state["reset_rng"][ids]
        if self._spawn_locations is not None and len(self._spawn_locations):
            sample, rng = self._uniform(rng, 1, joints.dtype)
            index = (sample[:, 0] * len(self._spawn_locations)).long()
            anchor = self._spawn_locations[index]
            root[:, :3] += anchor
            next_mdp_state["spawn_height"][ids] = anchor[:, 2:3]
        noise, rng = self._uniform(rng, 3 + joints.shape[-1], joints.dtype)
        next_mdp_state["reset_rng"][ids] = rng
        noise = 2 * noise - 1
        root[:, :2] += self.root_xy_noise * noise[:, :2]
        half_yaw = 0.5 * self.root_yaw_noise * noise[:, 2]
        sin_yaw, cos_yaw = half_yaw.sin(), half_yaw.cos()
        x, y, z, w = root[:, 3:7].unbind(dim=-1)
        root[:, 3:7] = torch.stack(
            (
                cos_yaw * x - sin_yaw * y,
                sin_yaw * x + cos_yaw * y,
                cos_yaw * z + sin_yaw * w,
                cos_yaw * w - sin_yaw * z,
            ),
            dim=-1,
        )
        lower, upper = self._joint_limits
        pose["joint_pos"] = (joints + self.joint_position_noise * noise[:, 3:]).clamp(
            min=lower[ids], max=upper[ids]
        )
        poses[self.entity] = pose
        return next_mdp_state, poses

    def _after_step(self, mdp_state, transition):
        transition["velocity_command"] = mdp_state["velocity_command"]
        steps_left = (mdp_state["command_steps_left"] - 1).clamp_min(0)
        mdp_state["command_steps_left"] = steps_left
        self._sample_command(mdp_state, (steps_left == 0) & ~transition["done"])

    def _sample_command(self, mdp_state, selected):
        previous = mdp_state["velocity_command"]
        samples, rng = self._uniform(mdp_state["command_rng"], 3, previous.dtype)
        lo = previous.new_tensor((self.vx[0], self.vy[0], self.yaw_rate[0]))
        hi = previous.new_tensor((self.vx[1], self.vy[1], self.yaw_rate[1]))
        mdp_state["command_rng"] = torch.where(selected, rng, mdp_state["command_rng"])
        mdp_state["velocity_command"] = torch.where(
            selected, lo + (hi - lo) * samples, previous
        )
        mdp_state["command_steps_left"] = torch.where(
            selected, self.hold_steps, mdp_state["command_steps_left"]
        )

    @staticmethod
    def _uniform(rng, count, dtype):
        # Park–Miller with one explicit RNG state per world.
        # TODO: is there a more efficient way to do this?
        samples = []
        for _ in range(count):
            rng = (rng * 48271) % 2147483647
            samples.append((rng >> 7).to(dtype) * (1.0 / (1 << 24)))
        return torch.cat(samples, dim=-1), rng
