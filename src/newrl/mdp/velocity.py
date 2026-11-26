"""Velocity tracking with task-owned command and physical reset sampling."""

from typing import Self, TYPE_CHECKING

from dataclasses import dataclass, field

import torch
import warp as wp
from tensordict import TensorDict

from newrl.utils.math import quat_from_yaw, quat_mul

from .base import MDP

if TYPE_CHECKING:
    from newrl.scene import Scene
    from newrl.visualizations.task import TaskVisualization


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

    def __post_init__(self) -> None:
        if self.hold_steps < 1:
            raise ValueError("hold_steps must be positive")
        if any(lo > hi for lo, hi in (self.vx, self.vy, self.yaw_rate)):
            raise ValueError("Command ranges must be ordered")
        if min(self.joint_position_noise, self.root_xy_noise, self.root_yaw_noise) < 0:
            raise ValueError("Reset noise amplitudes must be nonnegative")

    def _initialize(self, scene: "Scene") -> Self:
        bound = super()._initialize(scene)
        robot = scene.entities[self.entity]
        if robot.root_q_indices is None:
            raise ValueError("VelocityMDP requires a robot with one free root joint")
        bound._nominal_pose = TensorDict(
            {
                "root_pose": wp.to_torch(scene.model.joint_q)[robot.root_q_indices],
                "joint_pos": robot.default_position,
            },
            batch_size=[len(robot.joint_q_indices)],
            device=robot.joint_q_indices.device,
        )
        bound._spawn_locations = scene.spawn_locations
        bound._joint_limits = (
            wp.to_torch(scene.model.joint_limit_lower)[robot.joint_qd_indices],
            wp.to_torch(scene.model.joint_limit_upper)[robot.joint_qd_indices],
        )
        return bound

    def visualizations(self) -> tuple["TaskVisualization", ...]:
        from newrl.visualizations.velocity import VelocityVectors

        return (*super().visualizations(), VelocityVectors(self.entity))

    def initial_state(self, num_envs: int, action_dim: int, device: torch.device) -> TensorDict:
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

    @property
    def replay_fields(self) -> tuple[str, ...]:
        return (
            *super().replay_fields, "command_rng", "reset_rng",
            "command_steps_left", "velocity_command", "spawn_height",
        )

    def reset(
        self,
        mdp_state: TensorDict,
        ids: torch.Tensor | slice,
    ) -> tuple[TensorDict, dict[str, TensorDict]]:
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
        yaw = self.root_yaw_noise * noise[:, 2]
        root[:, 3:7] = quat_mul(quat_from_yaw(yaw), root[:, 3:7])
        lower, upper = self._joint_limits
        pose["joint_pos"] = (joints + self.joint_position_noise * noise[:, 3:]).clamp(
            min=lower[ids], max=upper[ids]
        )
        poses[self.entity] = pose
        return next_mdp_state, poses

    def _advance_task_state(
        self,
        physical_state: TensorDict,
        state: TensorDict,
    ) -> dict[str, torch.Tensor]:
        command = state["velocity_command"]
        steps_left = (state["command_steps_left"] - 1).clamp_min(0)
        state["command_steps_left"] = steps_left
        self._sample_command(state, steps_left == 0)
        return {"velocity_command": command}

    def _sample_command(self, mdp_state: TensorDict, selected: torch.Tensor) -> None:
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
    def _uniform(
        rng: torch.Tensor,
        count: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Park–Miller with one explicit RNG state per world.
        # TODO: is there a more efficient way to do this?
        samples = []
        for _ in range(count):
            rng = (rng * 48271) % 2147483647
            samples.append((rng >> 7).to(dtype) * (1.0 / (1 << 24)))
        return torch.cat(samples, dim=-1), rng
