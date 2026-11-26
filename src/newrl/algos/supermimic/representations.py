"""Generalized and maximal coordinates for differentiable dynamics."""

from typing import TYPE_CHECKING

from dataclasses import dataclass

import newton
import torch
from torch import nn
import warp as wp
from tensordict import TensorDict

from newrl.utils.config import cs
from newrl.utils.math import normalize, quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse, quat_to_6d

if TYPE_CHECKING:
    from newrl.scene import Scene


@dataclass(frozen=True)
class GeneralizedCoordinatesConfig:
    # Robot entity to encode and reconstruct.
    entity: str = "robot"
    # Body used as the free root and the local dynamics frame.
    root_body: str = "pelvis"
    # Loss scales for root position, orientation, linear and angular velocity, then joint
    # position and velocity.
    state_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 3.14, 0.5, 5.0)
    # Interpret pose predictions as direct increments or corrections to velocity drift.
    integration: str = "direct"

    def __post_init__(self) -> None:
        import math

        if self.integration not in ("direct", "velocity"):
            raise ValueError("integration must be direct or velocity")
        if len(self.state_scales) != 6 or any(not math.isfinite(v) or v <= 0 for v in self.state_scales):
            raise ValueError('Need six positive finite physical state scales')

    @property
    def physical_fields(self) -> tuple[tuple[str, str], ...]:
        return tuple((self.entity, name) for name in (
            "joint_pos", "joint_vel", "body_pose_w", "body_twist_w",
        ))

    def build(self, scene: "Scene") -> "GeneralizedCoordinates":
        return GeneralizedCoordinates(self, scene)


class _Coordinates(nn.Module):
    """Shared root and scalar joint state with link-origin world twists."""

    def __init__(self, config: GeneralizedCoordinatesConfig, scene: "Scene") -> None:
        super().__init__()
        self.config = config
        robot = scene.entities[config.entity]
        (self.root,), _ = robot.find_bodies(config.root_body)
        self.joint_names, self.body_names = robot.joint_names, robot.body_names
        self.register_buffer('com', wp.to_torch(scene.model.body_com)[robot.body_indices[0]].clone())
        self.feature_dim = 10 + 2 * len(self.joint_names)
        self.prediction_dim = 12 + 2 * len(self.joint_names)
        self.physical_fields = frozenset(config.physical_fields) | {(config.entity, 'projected_gravity')}
        self.control_dt = 0.0
        self.velocity_weight = 0.0
        if config.integration == "velocity":
            self.control_dt = scene.control_dt
            substeps = scene.config.substeps
            self.velocity_weight = (substeps + 1) / (2 * substeps)

    def _pose_step(
        self, pose: torch.Tensor, twist: torch.Tensor, prediction: torch.Tensor, frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dp, dr, dv, dw = prediction.split(3, dim=-1)
        velocity_delta = quat_rotate(frame, dv)
        angular_delta = quat_rotate(frame, dw)
        position_delta = quat_rotate(frame, dp)
        rotation_vector = quat_rotate(frame, dr)
        if self.config.integration == "velocity":
            position_delta = position_delta + self.control_dt * (twist[..., :3] + self.velocity_weight * velocity_delta)
            rotation_vector = rotation_vector + self.control_dt * (twist[..., 3:] + self.velocity_weight * angular_delta)
        angle = rotation_vector.norm(dim=-1, keepdim=True)
        delta_q = torch.cat((rotation_vector * (0.5 * torch.sinc(angle / (2 * torch.pi))), (angle / 2).cos()), -1)
        next_pose = torch.cat((pose[..., :3] + position_delta, normalize(quat_mul(delta_q, pose[..., 3:]))), -1)
        next_twist = twist + torch.cat((velocity_delta, angular_delta), -1)
        return next_pose, next_twist

    def _origin_twist(self, pose: torch.Tensor, twist: torch.Tensor, com: torch.Tensor) -> torch.Tensor:
        offset = quat_rotate(pose[..., 3:], com.expand_as(pose[..., :3]))
        linear = twist[..., :3] - torch.linalg.cross(twist[..., 3:], offset)
        return torch.cat((linear, twist[..., 3:]), -1)

    def _physical(self, state: TensorDict, pose: torch.Tensor, origin_twist: torch.Tensor) -> TensorDict:
        offset = quat_rotate(pose[..., 3:], self.com.expand_as(pose[..., :3]))
        omega = origin_twist[..., 3:]
        com_velocity = origin_twist[..., :3] + torch.linalg.cross(omega, offset)
        down = torch.zeros_like(pose[..., :3])
        down[..., 2] = -1
        return TensorDict({self.config.entity: TensorDict({
            'joint_pos': state['joint_pos'], 'joint_vel': state['joint_vel'],
            'body_pose_w': pose,
            'body_twist_w': torch.cat((com_velocity, omega), -1),
            'projected_gravity': quat_rotate_inverse(pose[..., 3:], down),
        }, batch_size=state.batch_size)}, batch_size=state.batch_size)

    def encode(self, physical_state: TensorDict) -> TensorDict:
        body = physical_state[self.config.entity]
        root = body['body_pose_w'][..., self.root, :]
        twist = body['body_twist_w'][..., self.root, :]
        return TensorDict({
            'root_pose': root.clone(),
            'root_twist': self._origin_twist(root, twist, self.com[self.root]),
            'joint_pos': body['joint_pos'].clone(), 'joint_vel': body['joint_vel'].clone(),
        }, batch_size=physical_state.batch_size)

    def features(self, state: TensorDict) -> torch.Tensor:
        root, twist = state['root_pose'], state['root_twist']
        down = torch.zeros_like(root[..., :3])
        down[..., 2] = -1
        return torch.cat((
            root[..., 2:3], quat_rotate_inverse(root[..., 3:], down),
            quat_rotate_inverse(root[..., 3:], twist[..., :3]),
            quat_rotate_inverse(root[..., 3:], twist[..., 3:]),
            state['joint_pos'], state['joint_vel'],
        ), -1)

    def apply_prediction(self, state: TensorDict, prediction: torch.Tensor) -> TensorDict:
        if prediction.shape != (*state.batch_size, self.prediction_dim):
            raise ValueError("Prediction shape does not match representation state")
        pose, twist = state['root_pose'], state['root_twist']
        pose, twist = self._pose_step(pose, twist, prediction[..., :12], pose[..., 3:])
        joints = len(self.joint_names)
        dq = prediction[..., 12:12 + joints]
        dqd = prediction[..., 12 + joints:12 + 2 * joints]
        if self.config.integration == "velocity":
            dq = dq + self.control_dt * (state['joint_vel'] + self.velocity_weight * dqd)
        return TensorDict({
            'root_pose': pose, 'root_twist': twist,
            'joint_pos': state['joint_pos'] + dq, 'joint_vel': state['joint_vel'] + dqd,
        }, batch_size=state.batch_size)

    def _pose_errors(self, pose: torch.Tensor, twist: torch.Tensor, target: torch.Tensor, target_twist: torch.Tensor) -> tuple[torch.Tensor, ...]:
        q = target[..., 3:]
        relative = quat_mul(quat_conjugate(q), pose[..., 3:])
        angle = 2 * torch.atan2(relative[..., :3].norm(dim=-1, keepdim=True), relative[..., 3:].abs())
        return (
            quat_rotate_inverse(q, pose[..., :3] - target[..., :3]),
            angle,
            quat_rotate_inverse(q, twist[..., :3] - target_twist[..., :3]),
            quat_rotate_inverse(q, twist[..., 3:] - target_twist[..., 3:]),
        )

    def _scaled_errors(self, predicted: TensorDict, target: TensorDict) -> torch.Tensor:
        errors = (*self._pose_errors(predicted['root_pose'], predicted['root_twist'], target['root_pose'], target['root_twist']),
                  predicted['joint_pos'] - target['joint_pos'], predicted['joint_vel'] - target['joint_vel'])
        return torch.cat([value / scale for value, scale in zip(errors, self.config.state_scales, strict=True)], -1)

    def compute_loss(self, predicted: TensorDict, target: TensorDict, valid: torch.Tensor) -> torch.Tensor:
        squared = self._scaled_errors(predicted, target.detach()).square().mean(-1, keepdim=True)
        return (squared * valid).sum() / valid.sum().clamp_min(1)


class GeneralizedCoordinates(_Coordinates):
    """One free-root tree with fixed, revolute and prismatic descendants.

    Compact root twists use world-frame link-origin linear/angular velocity.
    Decoded Newton body twists use world-frame COM linear/angular velocity.
    Runtime tensors preserve arbitrary leading batch dimensions and gradients.
    """

    def __init__(self, config: GeneralizedCoordinatesConfig, scene: "Scene") -> None:
        super().__init__(config, scene)
        robot, model = scene.entities[config.entity], scene.model
        if robot.root_q_indices is None:
            raise ValueError("Generalized coordinates require one free root")
        ids = robot.body_indices[0].cpu().tolist()
        local = {body: i for i, body in enumerate(ids)}
        children = wp.to_torch(model.joint_child).cpu().tolist()
        parents = wp.to_torch(model.joint_parent).cpu().tolist()
        kinds = wp.to_torch(model.joint_type).cpu().tolist()
        starts = wp.to_torch(model.joint_q_start).cpu().tolist()
        dofs = wp.to_torch(model.joint_qd_start).cpu().tolist()
        coordinates = {q: i for i, q in enumerate(robot.joint_q_indices[0].cpu().tolist())}
        incoming = {}
        for joint, child in enumerate(children):
            if child in local:
                if child in incoming:
                    raise ValueError("FK requires exactly one incoming joint per body")
                incoming[child] = joint
        if set(incoming) != set(ids):
            raise ValueError("FK requires a connected joint tree covering every robot body")
        root_joint = incoming[ids[self.root]]
        if kinds[root_joint] != newton.JointType.FREE or parents[root_joint] != -1:
            raise ValueError("The selected root must have a world-parent free joint")
        self.parents, self.kinds, self.coordinates = [], [], []
        axis = torch.zeros((len(ids), 3), device=robot.joint_q_indices.device)
        joint_ids = []
        for index, body in enumerate(ids):
            joint = incoming[body]
            kind = kinds[joint]
            if index != self.root and kind not in (
                newton.JointType.FIXED, newton.JointType.REVOLUTE, newton.JointType.PRISMATIC,
            ):
                raise ValueError("FK supports only fixed, revolute and prismatic descendants")
            if index != self.root and parents[joint] not in local:
                raise ValueError("Every descendant must belong to the selected free-root tree")
            self.parents.append(local.get(parents[joint], -1))
            self.kinds.append(kind)
            coordinate = coordinates.get(starts[joint], -1) if index != self.root else -1
            if kind in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                if coordinate < 0:
                    raise ValueError("Representation joint order does not cover all scalar joints")
                axis[index] = wp.to_torch(model.joint_axis)[dofs[joint]]
            self.coordinates.append(coordinate)
            joint_ids.append(joint)
        order = [self.root]
        levels = []
        while len(order) != len(ids):
            ready = [i for i in range(len(ids)) if i not in order and self.parents[i] in order]
            if not ready:
                raise ValueError("Robot joint tree contains a cycle or disconnected bodies")
            levels.append(tuple(ready))
            order.extend(ready)
        self.order = tuple(order)
        self.levels = tuple(levels)
        for depth, level in enumerate(self.levels):
            self.register_buffer(f'level_{depth}', torch.tensor(level, device=robot.joint_q_indices.device))
        self.register_buffer('coordinate_index', torch.tensor(self.coordinates, device=robot.joint_q_indices.device).clamp_min(0))
        self.register_buffer('revolute_mask', torch.tensor([kind == newton.JointType.REVOLUTE for kind in self.kinds], device=robot.joint_q_indices.device).float().unsqueeze(-1))
        self.register_buffer('prismatic_mask', torch.tensor([kind == newton.JointType.PRISMATIC for kind in self.kinds], device=robot.joint_q_indices.device).float().unsqueeze(-1))
        joint_ids = torch.tensor(joint_ids, device=robot.joint_q_indices.device)
        self.register_buffer('parent_frame', wp.to_torch(model.joint_X_p)[joint_ids].clone())
        self.register_buffer('child_frame', wp.to_torch(model.joint_X_c)[joint_ids].clone())
        self.register_buffer('axis', axis)

    def decode(self, state: TensorDict) -> TensorDict:
        """Reconstruct canonical body poses/twists and projected gravity in Torch."""
        poses, twists = {}, {}
        poses[self.root], twists[self.root] = state['root_pose'], state['root_twist']
        for depth, level in enumerate(self.levels):
            indices = getattr(self, f'level_{depth}')
            pp = torch.stack([poses[self.parents[index]] for index in level], -2)
            pt = torch.stack([twists[self.parents[index]] for index in level], -2)
            xp, xc = self.parent_frame[indices], self.child_frame[indices]
            anchor = pp[..., :3] + quat_rotate(pp[..., 3:], xp[..., :3].expand_as(pp[..., :3]))
            joint_q = quat_mul(pp[..., 3:], xp[..., 3:])
            axis = self.axis[indices].expand_as(pp[..., :3])
            axis_world = quat_rotate(joint_q, axis)
            position_value = state['joint_pos'].index_select(-1, self.coordinate_index[indices]).unsqueeze(-1)
            velocity_value = state['joint_vel'].index_select(-1, self.coordinate_index[indices]).unsqueeze(-1)
            half = position_value * self.revolute_mask[indices] / 2
            rotation = torch.cat((axis * half.sin(), half.cos()), -1)
            translation = axis * position_value * self.prismatic_mask[indices]
            angular = axis_world * velocity_value * self.revolute_mask[indices]
            linear = axis_world * velocity_value * self.prismatic_mask[indices]
            moving_q = quat_mul(joint_q, rotation)
            child_q = quat_mul(moving_q, quat_conjugate(xc[..., 3:]))
            position = anchor + quat_rotate(joint_q, translation) - quat_rotate(child_q, xc[..., :3].expand_as(translation))
            velocity = pt[..., :3] + torch.linalg.cross(pt[..., 3:], position - pp[..., :3]) + linear + torch.linalg.cross(angular, position - anchor)
            child_poses = torch.cat((position, child_q), -1).unbind(-2)
            child_twists = torch.cat((velocity, pt[..., 3:] + angular), -1).unbind(-2)
            for index, pose, twist in zip(level, child_poses, child_twists, strict=True):
                poses[index], twists[index] = pose, twist
        pose = torch.stack([poses[i] for i in range(len(self.body_names))], -2)
        origin_twist = torch.stack([twists[i] for i in range(len(self.body_names))], -2)
        return self._physical(state, pose, origin_twist)


@dataclass(frozen=True)
class MaximalCoordinatesConfig(GeneralizedCoordinatesConfig):
    """Predict body states alongside scalar joint coordinates."""

    def build(self, scene: "Scene") -> "MaximalCoordinates":
        return MaximalCoordinates(self, scene)


class MaximalCoordinates(_Coordinates):
    """Predict nonroot bodies directly, without FK or IK."""

    def __init__(self, config: MaximalCoordinatesConfig, scene: "Scene") -> None:
        super().__init__(config, scene)
        self._root_prediction_dim = self.prediction_dim
        self._other_bodies = len(self.body_names) - 1
        self.feature_dim += 15 * self._other_bodies
        self.prediction_dim += 12 * self._other_bodies

    def encode(self, physical_state: TensorDict) -> TensorDict:
        state = super().encode(physical_state)
        body = physical_state[self.config.entity]
        pose, twist = body['body_pose_w'], body['body_twist_w']
        origin_twist = self._origin_twist(pose, twist, self.com)
        state['link_pose'] = torch.cat((pose[..., :self.root, :], pose[..., self.root + 1:, :]), dim=-2)
        state['link_twist'] = torch.cat(
            (origin_twist[..., :self.root, :], origin_twist[..., self.root + 1:, :]), dim=-2
        )
        return state

    def features(self, state: TensorDict) -> torch.Tensor:
        root, pose, twist = state['root_pose'], state['link_pose'], state['link_twist']
        frame = root[..., None, 3:].expand_as(pose[..., 3:])
        local = torch.cat((
            quat_rotate_inverse(frame, pose[..., :3] - root[..., None, :3]),
            quat_to_6d(quat_mul(quat_conjugate(frame), pose[..., 3:])),
            quat_rotate_inverse(frame, twist[..., :3]),
            quat_rotate_inverse(frame, twist[..., 3:]),
        ), dim=-1)
        return torch.cat((super().features(state), local.flatten(-2)), dim=-1)

    def apply_prediction(self, state: TensorDict, prediction: torch.Tensor) -> TensorDict:
        result = super().apply_prediction(state, prediction)
        pose = state['link_pose']
        frame = state['root_pose'][..., None, 3:].expand_as(pose[..., 3:])
        link_prediction = prediction[..., self._root_prediction_dim:].reshape(
            *state.batch_size, self._other_bodies, 12
        )
        result['link_pose'], result['link_twist'] = self._pose_step(
            pose, state['link_twist'], link_prediction, frame,
        )
        return result

    def decode(self, state: TensorDict) -> TensorDict:
        pose, twist = state['link_pose'], state['link_twist']
        return self._physical(
            state,
            torch.cat((pose[..., :self.root, :], state['root_pose'].unsqueeze(-2), pose[..., self.root:, :]), dim=-2),
            torch.cat((twist[..., :self.root, :], state['root_twist'].unsqueeze(-2), twist[..., self.root:, :]), dim=-2),
        )

    def _scaled_errors(self, predicted: TensorDict, target: TensorDict) -> torch.Tensor:
        links = self._pose_errors(
            predicted['link_pose'], predicted['link_twist'], target['link_pose'], target['link_twist'],
        )
        scaled = torch.cat([
            error / scale for error, scale in zip(links, self.config.state_scales[:4], strict=True)
        ], dim=-1)
        return torch.cat((super()._scaled_errors(predicted, target), scaled.flatten(-2)), dim=-1)


cs.store("generalized", GeneralizedCoordinatesConfig(), group="representation")
cs.store("maximal", MaximalCoordinatesConfig(), group="representation")
