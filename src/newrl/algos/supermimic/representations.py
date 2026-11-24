"""Compact generalized coordinates and differentiable FK from Newton metadata."""

from dataclasses import dataclass

import newton
import torch
from torch import nn
import warp as wp
from tensordict import TensorDict

from newrl.utils.math import normalize, quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse
from newrl.utils.string import resolve_scene_names


@dataclass(frozen=True)
class GeneralizedCoordinatesConfig:
    entity: str = "robot"
    root_body: str = "pelvis"
    state_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 3.14, 0.5, 5.0)

    def __post_init__(self):
        import math

        if len(self.state_scales) != 6 or any(not math.isfinite(v) or v <= 0 for v in self.state_scales):
            raise ValueError('Need six positive finite physical state scales')

    @property
    def physical_fields(self):
        return tuple((self.entity, name) for name in (
            "joint_pos", "joint_vel", "body_pose_w", "body_twist_w",
        ))

    def build(self, scene):
        return GeneralizedCoordinates(self, scene)


class GeneralizedCoordinates(nn.Module):
    """One free-root tree with fixed, revolute and prismatic descendants.

    Compact root twists use world-frame link-origin linear/angular velocity.
    Decoded Newton body twists use world-frame COM linear/angular velocity.
    Runtime tensors preserve arbitrary leading batch dimensions and gradients.
    """

    def __init__(self, config, scene):
        super().__init__()
        self.config = config
        robot, model = scene.entities[config.entity], scene.model
        self.root, = resolve_scene_names(config.root_body, robot.body_names)
        if robot.root_q is None:
            raise ValueError("Generalized coordinates require one free root")
        self.joint_names, self.body_names = robot.joint_names, robot.body_names
        ids = robot.bodies[0].cpu().tolist()
        local = {body: i for i, body in enumerate(ids)}
        children = wp.to_torch(model.joint_child).cpu().tolist()
        parents = wp.to_torch(model.joint_parent).cpu().tolist()
        kinds = wp.to_torch(model.joint_type).cpu().tolist()
        starts = wp.to_torch(model.joint_q_start).cpu().tolist()
        dofs = wp.to_torch(model.joint_qd_start).cpu().tolist()
        coordinates = {q: i for i, q in enumerate(robot.q[0].cpu().tolist())}
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
        axis = torch.zeros((len(ids), 3), device=robot.q.device)
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
            self.register_buffer(f'level_{depth}', torch.tensor(level, device=robot.q.device))
        self.register_buffer('coordinate_index', torch.tensor(self.coordinates, device=robot.q.device).clamp_min(0))
        self.register_buffer('revolute_mask', torch.tensor([kind == newton.JointType.REVOLUTE for kind in self.kinds], device=robot.q.device).float().unsqueeze(-1))
        self.register_buffer('prismatic_mask', torch.tensor([kind == newton.JointType.PRISMATIC for kind in self.kinds], device=robot.q.device).float().unsqueeze(-1))
        joint_ids = torch.tensor(joint_ids, device=robot.q.device)
        self.register_buffer('parent_frame', wp.to_torch(model.joint_X_p)[joint_ids].clone())
        self.register_buffer('child_frame', wp.to_torch(model.joint_X_c)[joint_ids].clone())
        self.register_buffer('axis', axis)
        self.register_buffer('com', wp.to_torch(model.body_com)[robot.bodies[0]].clone())
        self.feature_dim = 10 + 2 * len(self.joint_names)
        self.prediction_dim = 12 + 2 * len(self.joint_names)
        self.physical_fields = frozenset(config.physical_fields) | {(config.entity, 'projected_gravity')}

    def encode(self, physical_state):
        body = physical_state[self.config.entity]
        root = body['body_pose_w'][..., self.root, :]
        twist = body['body_twist_w'][..., self.root, :]
        offset = quat_rotate(root[..., 3:], self.com[self.root].expand_as(root[..., :3]))
        origin_velocity = twist[..., :3] - torch.linalg.cross(twist[..., 3:], offset)
        return TensorDict({
            'root_pose': root.clone(),
            'root_twist': torch.cat((origin_velocity, twist[..., 3:]), -1),
            'joint_pos': body['joint_pos'].clone(), 'joint_vel': body['joint_vel'].clone(),
        }, batch_size=physical_state.batch_size)

    def features(self, state):
        root, twist = state['root_pose'], state['root_twist']
        down = torch.zeros_like(root[..., :3]); down[..., 2] = -1
        return torch.cat((
            root[..., 2:3], quat_rotate_inverse(root[..., 3:], down),
            quat_rotate_inverse(root[..., 3:], twist[..., :3]),
            quat_rotate_inverse(root[..., 3:], twist[..., 3:]),
            state['joint_pos'], state['joint_vel'],
        ), -1)

    def apply_prediction(self, state, prediction):
        """Direct position/twist increments in the current root frame; joints add."""
        if prediction.shape != (*state.batch_size, self.prediction_dim):
            raise ValueError("Prediction shape does not match representation state")
        pose, twist = state['root_pose'], state['root_twist']
        rotation_vector = prediction[..., 3:6]
        angle = rotation_vector.norm(dim=-1, keepdim=True)
        delta_q = torch.cat((rotation_vector * (0.5 * torch.sinc(angle / (2 * torch.pi))), (angle / 2).cos()), -1)
        joint_count = len(self.joint_names)
        return TensorDict({
            'root_pose': torch.cat((pose[..., :3] + quat_rotate(pose[..., 3:], prediction[..., :3]), normalize(quat_mul(pose[..., 3:], delta_q))), -1),
            'root_twist': torch.cat((twist[..., :3] + quat_rotate(pose[..., 3:], prediction[..., 6:9]), twist[..., 3:] + quat_rotate(pose[..., 3:], prediction[..., 9:12])), -1),
            'joint_pos': state['joint_pos'] + prediction[..., 12:12 + joint_count],
            'joint_vel': state['joint_vel'] + prediction[..., 12 + joint_count:],
        }, batch_size=state.batch_size)

    def decode(self, state):
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
        offset = quat_rotate(pose[..., 3:], self.com.expand_as(pose[..., :3]))
        com_velocity = origin_twist[..., :3] + torch.linalg.cross(origin_twist[..., 3:], offset)
        down = torch.zeros_like(pose[..., :3]); down[..., 2] = -1
        return TensorDict({self.config.entity: TensorDict({
            'joint_pos': state['joint_pos'], 'joint_vel': state['joint_vel'],
            'body_pose_w': pose,
            'body_twist_w': torch.cat((com_velocity, origin_twist[..., 3:]), -1),
            'projected_gravity': quat_rotate_inverse(pose[..., 3:], down),
        }, batch_size=state.batch_size)}, batch_size=state.batch_size)

    def compute_loss(self, predicted, target, valid):
        """Scaled multi-step state error; orientations are quaternion-sign invariant."""
        target = target.detach()
        q = target['root_pose'][..., 3:]
        relative = quat_mul(quat_conjugate(q), predicted['root_pose'][..., 3:])
        angle = 2 * torch.atan2(relative[..., :3].norm(dim=-1, keepdim=True), relative[..., 3:].abs())
        errors = (
            quat_rotate_inverse(q, predicted['root_pose'][..., :3] - target['root_pose'][..., :3]),
            angle,
            quat_rotate_inverse(q, predicted['root_twist'][..., :3] - target['root_twist'][..., :3]),
            quat_rotate_inverse(q, predicted['root_twist'][..., 3:] - target['root_twist'][..., 3:]),
            predicted['joint_pos'] - target['joint_pos'], predicted['joint_vel'] - target['joint_vel'],
        )
        squared = torch.cat([value / scale for value, scale in zip(errors, self.config.state_scales, strict=True)], -1).square().mean(-1, keepdim=True)
        return (squared * valid).sum() / valid.sum().clamp_min(1)
