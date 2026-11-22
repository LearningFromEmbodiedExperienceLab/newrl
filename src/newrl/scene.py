"""Replicate a Newton scene and expose named robot tensors."""

from dataclasses import dataclass

import newton
import torch
import warp as wp
from tensordict import TensorDict

from .contact import TerrainContact
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
        values = {}
        if "projected_gravity" in fields:
            pose = wp.to_torch(state.body_q)[self.bodies]
            axis = -pose[..., 3:6]
            gravity = torch.zeros_like(axis)
            gravity[..., 2] = -1.0
            cross = 2.0 * torch.linalg.cross(axis, gravity)
            values["projected_gravity"] = (
                gravity + pose[..., 6:7] * cross + torch.linalg.cross(axis, cross)
            )
            if "body_pose_w" in fields:
                values["body_pose_w"] = pose
        if "joint_torque" in fields:
            if observables is None or observables.qfrc_actuator is None:
                raise RuntimeError(
                    "Joint torque requires the solver actuator-force observable"
                )
            sources["joint_torque"] = (observables.qfrc_actuator, self.qd)
        return TensorDict(
            {
                name: values[name]
                if name in values
                else wp.to_torch(sources[name][0])[sources[name][1]]
                for name in fields
            },
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
            labels = [robot.joint_label[joint] for joint in scalar_joints(robot)]
            template.add_builder(robot, label_prefix=name)
            layouts[name] = (
                joints,
                bodies,
                roots,
                robot_cfg.action_scales(labels),
            )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, copy_count=config.num_envs)
        builder.default_shape_cfg.ke = config.ground_stiffness
        builder.default_shape_cfg.kd = config.ground_damping
        terrain_start = builder.shape_count
        spawn_locations = ()
        if isinstance(config.terrain, TerrainConfig):
            generated_spawns = config.terrain.build(builder)
            spawn_locations = (
                config.terrain.spawn_locations
                if generated_spawns is None
                else generated_spawns
            )
        elif callable(config.terrain):
            config.terrain(builder)
        elif config.terrain == "plane":
            builder.add_ground_plane()
        else:
            raise ValueError(
                "terrain must be a TerrainConfig, 'plane', or a Newton builder callback"
            )

        self.terrain_shapes = list(range(terrain_start, builder.shape_count))
        model = self.model = builder.finalize(device=config.device)
        device = wp.device_to_torch(model.device)
        self.spawn_locations = torch.as_tensor(
            spawn_locations, dtype=torch.float32, device=device
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
                action=JointPositionAction(
                    torch.tensor(scale, dtype=torch.float32, device=device)
                ),
            )

        self.contact_sensors = {}
        for name, sensor in config.contact_sensors.items():
            if name in self.entities:
                raise ValueError(f"Contact sensor name conflicts with entity {name!r}")
            self.contact_sensors[name] = TerrainContact(
                sensor, self.entities[sensor.entity]
            )

    def resolve_physical_fields(self, requested):
        """Validate entity fields and group them into a stable per-entity read plan."""
        supported = {
            "joint_pos",
            "joint_vel",
            "body_pose_w",
            "body_twist_w",
            "joint_torque",
            "projected_gravity",
        }
        fields_by_name = {name: tuple(sorted(supported)) for name in self.entities}
        fields_by_name.update(
            {name: sensor.fields for name, sensor in self.contact_sensors.items()}
        )
        if requested is None:
            return fields_by_name
        plan = {}
        for entity, field in sorted(requested):
            if entity not in fields_by_name:
                raise ValueError(f"Unknown physical entity {entity!r}")
            if field not in fields_by_name[entity]:
                raise ValueError(
                    f"Unknown physical field {field!r} for entity {entity!r}"
                )
            plan.setdefault(entity, []).append(field)
        return {name: tuple(fields) for name, fields in plan.items()}

    def read(self, state, plan=None, observables=None) -> TensorDict:
        if plan is None:
            plan = {
                name: fields
                for name, fields in self.resolve_physical_fields(None).items()
                if name in self.entities or self.contact_sensors[name].force is not None
            }
            if observables is None:
                plan = {
                    name: tuple(field for field in fields if field != "joint_torque")
                    for name, fields in plan.items()
                }
        return TensorDict(
            {
                name: self.entities[name].read(state, fields, observables)
                if name in self.entities
                else self.contact_sensors[name].read(fields)
                for name, fields in plan.items()
            },
            batch_size=[self.model.world_count],
            device=wp.device_to_torch(self.model.device),
        )
