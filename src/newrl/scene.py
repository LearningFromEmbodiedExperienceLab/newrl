"""Build batched Newton worlds and expose named entity tensors."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from tensordict import TensorDict

from .mdp.actions import ActionLayout
from .assets.objects import ObjectVariants
from .randomization import DomainRandomization
from .assets.robots.config import scalar_joints
from .sensors import FrameCamera, Sensor, ContactSensor
from .terrains import Terrain, TerrainConfig
from .utils.symmetry import SymmetryTransform
from .utils.string import resolve_scene_names

if TYPE_CHECKING:
    from newrl.configs import EnvironmentConfig
    from newrl.mdp.base import MDP


@dataclass(kw_only=True)
class Entity:
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_q_indices: torch.Tensor
    joint_qd_indices: torch.Tensor
    body_indices: torch.Tensor
    root_q_indices: torch.Tensor | None
    default_position: torch.Tensor
    joint_mirror: dict[str, tuple[str, int]] = field(default_factory=dict)
    body_mirror: dict[str, tuple[str, int]] = field(default_factory=dict)
    root_qd_indices: torch.Tensor | None = None
    model_parameters: dict[str, torch.Tensor] = field(default_factory=dict)

    def write_joint_state(
        self, state: newton.State, env_ids: torch.Tensor | slice, *,
        joint_pos: torch.Tensor | None = None, joint_vel: torch.Tensor | None = None,
    ) -> None:
        """Write selected worlds' scalar joint coordinates in entity order."""
        if joint_pos is not None:
            wp.to_torch(state.joint_q)[self.joint_q_indices[env_ids]] = joint_pos
        if joint_vel is not None:
            wp.to_torch(state.joint_qd)[self.joint_qd_indices[env_ids]] = joint_vel

    def write_root_state(
        self, state: newton.State, env_ids: torch.Tensor | slice, *,
        root_pose: torch.Tensor | None = None, root_twist: torch.Tensor | None = None,
    ) -> None:
        """Write world XYZ/XYZW pose and world COM linear/angular velocity."""
        if (root_pose is not None and self.root_q_indices is None) or (
            root_twist is not None and self.root_qd_indices is None
        ):
            raise ValueError("Writing root state requires an entity with one free root")
        if root_pose is not None:
            wp.to_torch(state.joint_q)[self.root_q_indices[env_ids]] = root_pose
        if root_twist is not None:
            wp.to_torch(state.joint_qd)[self.root_qd_indices[env_ids]] = root_twist

    def find_bodies(
        self, names: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Return entity-local body indices and labels matching the queries."""
        indices = resolve_scene_names(names, self.body_names, preserve_order)
        return indices, [self.body_names[index] for index in indices]

    def find_joints(
        self, names: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Return entity-local scalar joint indices and labels matching the queries."""
        indices = resolve_scene_names(names, self.joint_names, preserve_order)
        return indices, [self.joint_names[index] for index in indices]

    def joint_symmetry_transform(self) -> SymmetryTransform:
        return SymmetryTransform.from_names(self.joint_names, self.joint_mirror)

    def body_symmetry_transform(
        self, indices: Sequence[int], component_signs: tuple[int, ...]
    ) -> SymmetryTransform:
        labels = [self.body_names[index] for index in indices]
        return SymmetryTransform.from_names(labels, self.body_mirror, component_signs)

    def read(
        self,
        state: newton.State,
        fields: Sequence[str],
        observables: newton.solvers.SolverMuJoCo.Observables | SimpleNamespace | None = None,
    ) -> TensorDict:
        # Advanced indexing produces snapshots, safe across state-buffer swaps.
        sources = {
            "joint_pos": (state.joint_q, self.joint_q_indices),
            "joint_vel": (state.joint_qd, self.joint_qd_indices),
            "body_pose_w": (state.body_q, self.body_indices),
            "body_twist_w": (state.body_qd, self.body_indices),
        }
        values = {
            name: self.model_parameters[name]
            for name in fields
            if name in self.model_parameters
        }
        if "projected_gravity" in fields:
            pose = wp.to_torch(state.body_q)[self.body_indices]
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
            sources["joint_torque"] = (observables.qfrc_actuator, self.joint_qd_indices)
        return TensorDict(
            {
                name: values[name]
                if name in values
                else wp.to_torch(sources[name][0])[sources[name][1]]
                for name in fields
            },
            batch_size=[len(self.joint_q_indices)],
            device=self.joint_q_indices.device,
        )


@dataclass(kw_only=True)
class RobotEntity(Entity):
    joint_target_indices: torch.Tensor

    def write_joint_target(
        self, control: newton.Control, env_ids: torch.Tensor | slice, *,
        joint_pos: torch.Tensor | float | None = None,
        joint_vel: torch.Tensor | float | None = None,
    ) -> None:
        """Write position and/or velocity targets in entity order for the selected worlds."""
        if joint_pos is not None:
            wp.to_torch(control.joint_target_q)[self.joint_target_indices[env_ids]] = joint_pos
        if joint_vel is not None:
            wp.to_torch(control.joint_target_qd)[self.joint_qd_indices[env_ids]] = joint_vel


class Scene:
    physical_read_plan: dict[str, tuple[str, ...]]

    def __init__(self, config: "EnvironmentConfig", mdp: "MDP | None" = None) -> None:
        self.config = config
        self.sim_dt = config.sim_dt
        self.control_dt = config.sim_dt * config.substeps
        overlap = config.robots.keys() & config.objects.keys()
        if overlap:
            raise ValueError(f"Robot and object names overlap: {sorted(overlap)}")
        robots = {name: factory() for name, factory in config.robots.items()}
        components = {
            name: [cfg.build(wheel_mesh_segments=32 if config.collision == "mujoco" else None)]
            for name, cfg in robots.items()
        }
        robot_joints = {name: scalar_joints(parts[0]) for name, parts in components.items()}
        self.actions = ActionLayout(
            (config.mdp if mdp is None else mdp).actions,
            {name: (parts[0], robot_joints[name]) for name, parts in components.items()},
            self.sim_dt,
        )
        rng = Random(config.object_seed)
        self.object_variant_ids: dict[str, tuple[int, ...]] = {}
        for name, cfg in config.objects.items():
            if isinstance(cfg, ObjectVariants):
                options = [variant.build() for variant in cfg.variants]
                for index, option in enumerate(options[1:], 1):
                    for attr in ("body_label", "joint_label", "joint_type", "joint_parent",
                                 "joint_child", "joint_dof_dim", "articulation_start"):
                        if getattr(options[0], attr) != getattr(option, attr):
                            raise ValueError(f"Object {name!r} variant {index} differs in {attr}")
                self.object_variant_ids[name] = tuple(
                    rng.choices(range(len(options)), weights=cfg.weights, k=config.num_envs)
                )
            else:
                options = [cfg.build()]
            if any(not option.body_count for option in options):
                raise ValueError(f"Object {name!r} must have a rigid body")
            components[name] = options

        layouts = {}
        templates: dict[tuple[int, ...], newton.ModelBuilder] = {}

        def world_template(selection: tuple[int, ...]) -> newton.ModelBuilder:
            if selection in templates:
                return templates[selection]
            world = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverMuJoCo.register_custom_attributes(world)
            chosen = dict(zip(self.object_variant_ids, selection, strict=True))
            for name, options in components.items():
                part = options[chosen.get(name, 0)]
                if name not in layouts:
                    local_joints = robot_joints[name] if name in robot_joints else scalar_joints(part)
                    joints = [world.joint_count + j for j in local_joints]
                    bodies = range(world.body_count, world.body_count + part.body_count)
                    roots = [world.joint_count + j for j, kind in enumerate(part.joint_type)
                             if kind == newton.JointType.FREE and part.joint_parent[j] == -1]
                    layouts[name] = joints, bodies, roots
                world.add_builder(part, label_prefix=name)
            templates[selection] = world
            return world

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.object_variant_ids:
            for selection in zip(*self.object_variant_ids.values(), strict=True):
                builder.add_world(world_template(selection))
        else:
            builder.replicate(world_template(()), copy_count=config.num_envs)
        template = next(iter(templates.values()))
        builder.default_shape_cfg.ke = config.ground_stiffness
        builder.default_shape_cfg.kd = config.ground_damping
        terrain_start = builder.shape_count
        self.terrain = Terrain()
        if isinstance(config.terrain, TerrainConfig):
            built = config.terrain.build(builder)
            if isinstance(built, Terrain):
                self.terrain = built
            else:
                self.terrain = Terrain(
                    config.terrain.spawn_locations if built is None else built
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
            self.terrain.spawn_locations, dtype=torch.float32, device=device
        ).reshape(-1, 3)
        joint_offsets = wp.to_torch(model.joint_world_start)[:config.num_envs, None].long()
        body_offsets = wp.to_torch(model.body_world_start)[:config.num_envs, None].long()
        self.entities: dict[str, Entity] = {}
        self.robots: dict[str, RobotEntity] = {}
        for name, (joints, bodies, roots) in layouts.items():
            joint_ids = joint_offsets + torch.tensor(joints, device=device, dtype=torch.long)
            body_ids = body_offsets + torch.tensor(list(bodies), device=device, dtype=torch.long)
            joint_q_indices = wp.to_torch(model.joint_q_start)[joint_ids].long()
            joint_qd_indices = wp.to_torch(model.joint_qd_start)[joint_ids].long()
            root_q_indices = None
            root_qd_indices = None
            if len(roots) == 1:
                root_joint_ids = joint_offsets + roots[0]
                root_start = wp.to_torch(model.joint_q_start)[root_joint_ids].long()
                root_q_indices = root_start + torch.arange(7, device=device)
                root_dof = wp.to_torch(model.joint_qd_start)[root_joint_ids].long()
                root_qd_indices = root_dof + torch.arange(6, device=device)
            labels = tuple(template.joint_label[j] for j in joints)
            fields = dict(
                joint_names=labels,
                body_names=tuple(template.body_label[b] for b in bodies),
                joint_q_indices=joint_q_indices,
                joint_qd_indices=joint_qd_indices,
                body_indices=body_ids,
                root_q_indices=root_q_indices,
                root_qd_indices=root_qd_indices,
                default_position=wp.to_torch(model.joint_q)[joint_q_indices],
            )
            if name in robots:
                robot = robots[name]
                self.robots[name] = RobotEntity(
                    **fields,
                    joint_target_indices=wp.to_torch(model.joint_target_q_start)[joint_ids].long(),
                    joint_mirror=robot.joint_mirror, body_mirror=robot.body_mirror,
                )
                self.entities[name] = self.robots[name]
            else:
                self.entities[name] = Entity(**fields)

        self.actions._initialize(self)
        self.randomization: DomainRandomization | None = None
        self.sensors: dict[str, Sensor] = {}
        for name, sensor in config.sensors.items():
            if name in self.entities:
                raise ValueError(f"Sensor name conflicts with entity {name!r}")
            self.sensors[name] = sensor.build(self)

    def _initialize(self, mdp: "MDP") -> "MDP":
        """Bind the MDP and initialize the scene reads it and the caller need."""
        if (config := self.config.randomization) is not None:
            self.randomization = DomainRandomization(self, config)

        mdp = mdp._initialize(self)
        outputs = frozenset(self.config.output_physical_fields)
        self.resolve_physical_fields(outputs)
        self.output_physical_fields = tuple(sorted(outputs))
        required = mdp.physical_fields
        self.physical_read_plan = self.resolve_physical_fields(
            None if required is None else required | outputs
        )
        self.needs_torque = any(
            "joint_torque" in self.physical_read_plan.get(name, ())
            for name in self.entities
        )
        self.contact_sensors: dict[str, ContactSensor] = {}
        self.cameras: dict[str, FrameCamera] = {}
        for name, sensor in self.sensors.items():
            if name in self.physical_read_plan:
                sensor.initialize(self)
                if isinstance(sensor, ContactSensor):
                    self.contact_sensors[name] = sensor
                if isinstance(sensor, FrameCamera):
                    self.cameras[name] = sensor
        self.camera_bvh_refit = any(camera.requires_bvh_refit for camera in self.cameras.values())
        return mdp

    def enable_camera_previews(self) -> None:
        """Activate configured frame cameras for playback before the first reset."""
        for name, sensor in self.sensors.items():
            if name not in self.cameras and isinstance(sensor, FrameCamera):
                sensor.initialize(self)
                self.cameras[name] = sensor
                self.camera_bvh_refit |= sensor.requires_bvh_refit

    def action_symmetry_transform(self) -> SymmetryTransform:
        """Build a CPU transform in flattened policy channel order."""
        return self.actions.symmetry_transform()

    def resolve_physical_fields(
        self,
        requested: Iterable[tuple[str, str]] | None,
    ) -> dict[str, tuple[str, ...]]:
        """Validate entity fields and group them into a stable per-entity read plan."""
        supported = {
            "joint_pos",
            "joint_vel",
            "body_pose_w",
            "body_twist_w",
            "projected_gravity",
        }
        fields_by_name = {
            name: tuple(sorted(supported | entity.model_parameters.keys() | ({"joint_torque"} if name in self.robots else set())))
            for name, entity in self.entities.items()
        }
        fields_by_name.update(
            {name: sensor.fields for name, sensor in self.sensors.items()}
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

    def read(
        self,
        state: newton.State,
        observables: newton.solvers.SolverMuJoCo.Observables | SimpleNamespace | None,
        *,
        field_plan: dict[str, tuple[str, ...]] | None = None,
    ) -> TensorDict:
        values = {}
        plan = self.physical_read_plan if field_plan is None else field_plan
        for name, fields in plan.items():
            if name in self.entities:
                values[name] = self.entities[name].read(state, fields, observables)
            else:
                values[name] = self.sensors[name].read(state, fields)
        return TensorDict(
            values,
            batch_size=[self.model.world_count],
            device=wp.device_to_torch(self.model.device),
        )
