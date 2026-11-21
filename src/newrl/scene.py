"""Replicate a Newton scene and expose named robot tensors."""

from dataclasses import dataclass

import newton
import torch
import warp as wp
from tensordict import TensorDict

from .mdp.actions import JointPositionAction
from .robots import scalar_joints
from .terrain import TerrainConfig


@dataclass
class RobotEntity:
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    q: torch.Tensor
    qd: torch.Tensor
    targets: torch.Tensor
    bodies: torch.Tensor
    root_q: torch.Tensor | None
    default_position: torch.Tensor
    action: JointPositionAction

    def read(self, state, fields, observables=None) -> TensorDict:
        # Advanced indexing produces snapshots, safe across state-buffer swaps.
        sources = {
            "joint_pos": (state.joint_q, self.q),
            "joint_vel": (state.joint_qd, self.qd),
            "body_pose_w": (state.body_q, self.bodies),
            "body_twist_w": (state.body_qd, self.bodies),
        }
        if "joint_torque" in fields:
            if observables is None or observables.qfrc_actuator is None:
                raise RuntimeError(
                    "Joint torque requires the solver actuator-force observable"
                )
            sources["joint_torque"] = (observables.qfrc_actuator, self.qd)
        return TensorDict(
            {name: wp.to_torch(sources[name][0])[sources[name][1]] for name in fields},
            batch_size=[len(self.q)],
            device=self.q.device,
        )

    def write_action(self, control, action: torch.Tensor) -> None:
        wp.to_torch(control.joint_target_q)[self.targets] = self.action(
            action, self.default_position
        )


class Scene:
    def __init__(self, config):
        self.sim_dt = config.sim_dt
        self.control_dt = config.sim_dt * config.substeps
        template = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        layouts = {}
        for name, factory in config.robots.items():
            robot_cfg = factory()
            robot = robot_cfg.build()
            joints = [template.joint_count + j for j in scalar_joints(robot)]
            bodies = range(template.body_count, template.body_count + robot.body_count)
            roots = [
                template.joint_count + j
                for j, kind in enumerate(robot.joint_type)
                if kind == newton.JointType.FREE
            ]
            template.add_builder(robot, label_prefix=name)
            layouts[name] = (joints, bodies, roots, robot_cfg.action_scale)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, copy_count=config.num_envs)
        builder.default_shape_cfg.ke = config.ground_stiffness
        builder.default_shape_cfg.kd = config.ground_damping
        if isinstance(config.terrain, TerrainConfig):
            config.terrain.build(builder)
        elif callable(config.terrain):
            config.terrain(builder)
        elif config.terrain == "plane":
            builder.add_ground_plane()
        else:
            raise ValueError(
                "terrain must be a TerrainConfig, 'plane', or a Newton builder callback"
            )

        model = self.model = builder.finalize(device=config.device)
        device = wp.device_to_torch(model.device)
        self.spawn_locations = torch.as_tensor(
            config.terrain.spawn_locations
            if isinstance(config.terrain, TerrainConfig)
            else (),
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 3)
        worlds = torch.arange(config.num_envs, device=device)[:, None]
        self.entities: dict[str, RobotEntity] = {}
        for name, (joints, bodies, roots, scale) in layouts.items():
            joint_ids = worlds * template.joint_count + torch.tensor(
                joints, device=device, dtype=torch.long
            )
            body_ids = worlds * template.body_count + torch.tensor(
                list(bodies), device=device, dtype=torch.long
            )
            q = wp.to_torch(model.joint_q_start)[joint_ids].long()
            qd = wp.to_torch(model.joint_qd_start)[joint_ids].long()
            targets = wp.to_torch(model.joint_target_q_start)[joint_ids].long()
            root_q = None
            if len(roots) == 1:
                root_joint_ids = worlds * template.joint_count + roots[0]
                root_start = wp.to_torch(model.joint_q_start)[root_joint_ids].long()
                root_q = root_start + torch.arange(7, device=device)
            self.entities[name] = RobotEntity(
                joint_names=tuple(template.joint_label[j] for j in joints),
                body_names=tuple(template.body_label[b] for b in bodies),
                q=q,
                qd=qd,
                targets=targets,
                bodies=body_ids,
                root_q=root_q,
                default_position=wp.to_torch(model.joint_q)[q],
                action=JointPositionAction(scale),
            )

    def resolve_physical_fields(self, requested):
        """Validate entity fields and group them into a stable per-entity read plan."""
        supported = {
            "joint_pos",
            "joint_vel",
            "body_pose_w",
            "body_twist_w",
            "joint_torque",
        }
        if requested is None:
            return {name: tuple(sorted(supported)) for name in self.entities}
        plan = {}
        for entity, field in sorted(requested):
            if entity not in self.entities:
                raise ValueError(f"Unknown physical entity {entity!r}")
            if field not in supported:
                raise ValueError(
                    f"Unknown physical field {field!r} for entity {entity!r}"
                )
            plan.setdefault(entity, []).append(field)
        return {name: tuple(fields) for name, fields in plan.items()}

    def read(self, state, plan=None, observables=None) -> TensorDict:
        if plan is None:
            plan = self.resolve_physical_fields(None)
            if observables is None:
                plan = {
                    name: tuple(field for field in fields if field != "joint_torque")
                    for name, fields in plan.items()
                }
        return TensorDict(
            {
                name: self.entities[name].read(state, fields, observables)
                for name, fields in plan.items()
            },
            batch_size=[self.model.world_count],
            device=wp.device_to_torch(self.model.device),
        )
