"""Heading-aligned velocity command and measured planar velocity arrows."""

from dataclasses import dataclass, replace

import newton
import torch
import warp as wp
from tensordict import TensorDict

from newrl.scene import Scene
from newrl.utils.math import quat_rotate, yaw_quat

from .markers import Marker


@dataclass(frozen=True)
class VelocityVectors:
    entity: str
    _base: int = 0

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.entity, "body_pose_w"), (self.entity, "body_twist_w")})

    @property
    def mdp_fields(self) -> tuple[str, ...]:
        return ("velocity_command",)

    @property
    def arrows(self) -> tuple[Marker, ...]:
        return tuple(
            Marker(name, "arrow", (0, 0, 0, 0, 0, 0, 1), size=0.65, color=color, group=label)
            for name, color, label in (
                ("velocity/command", 0x40C0FF, "Commanded velocity"),
                ("velocity/actual", 0xFFBA68, "Actual velocity"),
            )
        )

    def _initialize(self, scene: Scene) -> "VelocityVectors":
        entity = scene.entities[self.entity]
        model = scene.model
        root_start = entity.root_q_indices[0, 0]
        starts = wp.to_torch(model.joint_q_start)[:model.joint_count]
        root_joint = (starts == root_start) & (wp.to_torch(model.joint_type) == int(newton.JointType.FREE))
        (base,) = wp.to_torch(model.joint_child)[root_joint].tolist()
        (slot,) = (entity.body_indices[0] == base).nonzero(as_tuple=True)[0].tolist()
        return replace(self, _base=slot)

    def sample(self, physical_state: TensorDict, mdp_state: TensorDict) -> dict[str, torch.Tensor]:
        pose = physical_state[self.entity, "body_pose_w"][:, self._base]
        twist = physical_state[self.entity, "body_twist_w"][:, self._base]
        command = mdp_state["velocity_command"][:, :2]
        command = torch.cat((command, torch.zeros_like(command[:, :1])), dim=-1)
        command = quat_rotate(yaw_quat(pose[:, 3:]), command)
        actual = torch.cat((twist[:, :2], torch.zeros_like(twist[:, :1])), dim=-1)
        # Separate the origins in Z so matching arrows remain distinguishable.
        above = pose.new_tensor((0, 0, 0.28))
        below = pose.new_tensor((0, 0, 0.16))
        return {
            "velocity/command": torch.cat((pose[:, :3] + above, command), dim=-1),
            "velocity/actual": torch.cat((pose[:, :3] + below, actual), dim=-1),
        }
