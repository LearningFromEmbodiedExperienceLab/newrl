"""Scene, simulation, and task configuration for the batched environment."""

from typing import Any

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import newton

from .assets.objects import BoxConfig, CylinderConfig, ObjectConfig, ObjectVariants, SphereConfig, TableConfig
from .mdp.actions import ActionConfig, JointPosition, JointVelocity
from .assets.robots.g1 import g1_actuators
from .mdp.base import MDP, ObservationGroup, RewardGroup
from .mdp.episode_ends import HeightTermination, TiltTermination, Timeout
from .mdp.observations import (
    BodyAngularVelocity,
    BodyHeight,
    BodyLinearVelocity,
    ContactObservation,
    EntityField,
    ProjectedGravity,
    previous_action,
    velocity_command,
)
from .mdp.rewards import (
    Survival,
    ActionRate,
    FeetAirTime,
    FootSlip,
    JointAcc2,
    JointEnergy1,
    JointVel2,
    TrackVelocity,
    BodyAngularVelocity2,
    BodyLinearVelocity2,
)
from .mdp.velocity import VelocityMDP
from .mdp.motion_tracking import MotionTrackingMDP
from .mdp.episode_ends import ClipEnd, TrackingDeviation
from .mdp.observations import ReferenceMotion
from .mdp.rewards import JointLimit, TrackBodyMotion
from .randomization import BodyMassScale, DomainRandomizationConfig, JointGainScale, ShapeFriction
from .assets.robots import (
    RobotConfig, assetx_as2, assetx_as2w, assetx_as2w_sphere,
    assetx_m20s, assetx_spot, newton_anymal_c,
    assetx_g1,
)
from .sensors import SensorConfig, ContactSensorConfig
from .solver_adapters import MujocoSimConfig, SteelSimConfig
from .terrains import (
    BorderConfig,
    FlatTileConfig,
    GeneratedTerrainConfig,
    PlaneTerrainConfig,
    PyramidSlopeTileConfig,
    RandomGridTileConfig,
    ReconstructedTerrainConfig,
    RoughHeightfieldTileConfig,
    SteppingStonesTileConfig,
    StairsTileConfig,
    TerrainConfig,
)


@dataclass
class EnvironmentConfig:
    num_envs: int = 1
    device: str = "cuda:0"
    sim_dt: float = 0.005
    substeps: int = 4
    solver: MujocoSimConfig | SteelSimConfig = field(default_factory=MujocoSimConfig)
    ground_stiffness: float = 1.0e3
    ground_damping: float = 1.0e2
    graph_capture: bool = True
    deterministic: bool = False
    terrain: TerrainConfig | str | Callable[[newton.ModelBuilder], None] = field(
        default_factory=PlaneTerrainConfig
    )
    robots: dict[str, Callable[[], RobotConfig]] = field(
        default_factory=dict, metadata={"item_preset_group": "robot"}
    )
    objects: dict[str, ObjectConfig | ObjectVariants] = field(default_factory=dict)
    object_seed: int = 0
    mdp: MDP = field(default_factory=MDP)
    output_physical_fields: tuple[tuple[str, str], ...] = ()
    output_mdp_fields: tuple[str, ...] = ()
    name: str = "custom"
    collision: str = "newton"
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    randomization: DomainRandomizationConfig | None = None

    def __post_init__(self) -> None:
        if self.collision not in ("newton", "mujoco", "newrl"):
            raise ValueError("collision must be newton, mujoco, or newrl")


def _g1_actions() -> dict[str, ActionConfig]:
    scales = {
        pattern: 0.25 * group.effort_limit / group.stiffness
        for group in g1_actuators()
        for pattern in group.joint_names
    }
    return {"position": JointPosition(scale=scales)}


def locomotion_mdp(mdp_type: type[MDP] = MDP, base: str = "base") -> MDP:
    return mdp_type(
        observations={
            "proprio": ObservationGroup(
                {
                    "joint_pos": EntityField("robot", "joint_pos"),
                    "joint_vel": EntityField("robot", "joint_vel"),
                    "projected_gravity": ProjectedGravity("robot", base),
                    "previous_action": previous_action,
                    "root_ang_vel_b": BodyAngularVelocity(
                        "robot", base, frame="body"
                    ),
                }
            ),
            "privileged": ObservationGroup(
                {
                    "root_lin_vel_b": BodyLinearVelocity("robot", base, frame="body"),
                    "root_height": BodyHeight("robot", base),
                }
            ),
        },
        # rewards={"survival": RewardGroup({"survival": Survival(weight=1.0)})},
        truncations={"timeout": Timeout(1000)},
    )


def G1Tabletop() -> EnvironmentConfig:
    """A survival-only example with one passive tabletop object per world."""
    x, y = 0.85, 0.0
    table = TableConfig(position=(x, y, 0))
    top = table.height
    return EnvironmentConfig(
        name="G1Tabletop",
        num_envs=4,
        # Full-body contacts can exhaust the default constraint row capacity.
        solver=SteelSimConfig(articulation_constraint_max=512),
        robots={"robot": assetx_g1},
        objects={
            "table": table,
            "object": ObjectVariants((
                BoxConfig(half_extents=(0.09, 0.06, 0.04), position=(x, y, top + 0.04)),
                SphereConfig(radius=0.065, position=(x, y, top + 0.065)),
                CylinderConfig(radius=0.05, half_height=0.10, position=(x, y, top + 0.10)),
            )),
        },
        mdp=MDP(
            actions=_g1_actions(),
            observations={
                "proprio": ObservationGroup({"joint_pos": EntityField("robot", "joint_pos")}),
                "privileged": ObservationGroup({"object_pose": EntityField("object", "body_pose_w")}),
            },
            rewards={"survival": RewardGroup({"survival": Survival(weight=1.0)})},
            truncations={"timeout": Timeout(1000)},
        ),
    )


def G1MotionTracking(**overrides: Any) -> EnvironmentConfig:
    """50 Hz G1 tracking of the official retargeted LAFAN1 library."""
    bodies = (
        "pelvis", "torso_link",
        "(left|right)_hip_roll_link",
        "(left|right)_knee_link",
        "(left|right)_ankle_roll_link",
        "(left|right)_shoulder_roll_link",
        "(left|right)_elbow_link",
        "(left|right)_wrist_yaw_link",
    )
    mdp = MotionTrackingMDP(
        actions=_g1_actions(),
        observations={
            "proprio": ObservationGroup(
                {
                    "joint_pos": EntityField("robot", "joint_pos"),
                    "joint_vel": EntityField("robot", "joint_vel"),
                    "projected_gravity": ProjectedGravity("robot", "pelvis"),
                    "previous_action": previous_action,
                    "root_ang_vel_b": BodyAngularVelocity("robot", "pelvis", frame="body"),
                }
            ),
            "privileged": ObservationGroup(
                {
                    "root_lin_vel_b": BodyLinearVelocity("robot", "pelvis", frame="body"),
                    "root_height": BodyHeight("robot", "pelvis"),
                }
            ),
            "command": ObservationGroup({"reference": ReferenceMotion()}),
        },
        rewards={
            "tracking": RewardGroup({
                "body_pos": TrackBodyMotion(1.0, 0.3, "position", body_names=bodies, anchor_body="pelvis"),
                "body_ori": TrackBodyMotion(1.0, 0.4, "orientation", body_names=bodies, anchor_body="pelvis"),
                "body_lin_vel": TrackBodyMotion(1.0, 1.0, "linear_velocity", body_names=bodies),
                "body_ang_vel": TrackBodyMotion(1.0, 3.14, "angular_velocity", body_names=bodies),
                "root_pos": TrackBodyMotion(0.5, 0.3, body_names="pelvis"),
                "root_ori": TrackBodyMotion(0.5, 0.4, "orientation", body_names="pelvis"),
                "action_rate": ActionRate(0.05),
                "joint_limit": JointLimit(2.0),
            }),
        },
        terminations={
            "height": TrackingDeviation(BodyHeight("robot", "pelvis")),
            "tilt": TrackingDeviation(BodyHeight("robot", "pelvis"), "tilt", 0.8),
        },
        truncations={"timeout": Timeout(1000), "clip_end": ClipEnd()},
    )
    options = dict(num_envs=4096, robots={"robot": assetx_g1}, mdp=mdp, name="G1LAFAN1Tracking")
    options.update(overrides)
    return EnvironmentConfig(**options)


def G1Locomotion(
    *,
    contact_observation: bool = False,
    foot_slip_weight: float = 0.1,
    **overrides: Any,
) -> EnvironmentConfig:
    """Track planar velocity and yaw rate with the 29-DoF Unitree G1."""
    config = _velocity_locomotion(
        assetx_g1,
        actions=_g1_actions(),
        base="pelvis",
        feet=r"(left|right)_ankle_roll_link",
        contact_observation=contact_observation,
        foot_slip_weight=foot_slip_weight,
        name="G1LocoFlat",
    )
    config.mdp.terminations["height"] = HeightTermination(
        BodyHeight("robot", "pelvis"), min_height=0.45
    )
    config.mdp.rewards["survival"] = RewardGroup({"survival": Survival(weight=1.0)})
    regularization = config.mdp.rewards["regularization"].terms
    config.mdp.rewards["regularization"] = RewardGroup(
        (*regularization, ("joint_limit", JointLimit(weight=2.0)))
    )
    if sensors := overrides.pop("sensors", None):
        config.sensors.update(sensors)
    return replace(config, **overrides)


def AnymalLocomotion(
    *,
    contact_observation: bool = False,
    foot_slip_weight: float = 0.1,
    **overrides: Any,
) -> EnvironmentConfig:
    return _velocity_locomotion(
        newton_anymal_c,
        actions={"position": JointPosition(scale=0.5)},
        base="base",
        feet=r"(LF|LH|RF|RH)_FOOT",
        contact_observation=contact_observation,
        foot_slip_weight=foot_slip_weight,
        **overrides,
    )


def SpotLocomotion(
    *,
    contact_observation: bool = False,
    foot_slip_weight: float = 0.1,
    **overrides: Any,
) -> EnvironmentConfig:
    return _velocity_locomotion(
        assetx_spot,
        actions={"position": JointPosition(scale=0.5)},
        base="base_link",
        feet=".*_foot",
        contact_observation=contact_observation,
        foot_slip_weight=foot_slip_weight,
        **overrides,
    )


def AS2Locomotion(
    *,
    contact_observation: bool = False,
    foot_slip_weight: float = 0.1,
    **overrides: Any,
) -> EnvironmentConfig:
    return _velocity_locomotion(
        assetx_as2,
        actions={"position": JointPosition(scale=0.4)},
        base="base_link",
        feet=".*_foot",
        contact_observation=contact_observation,
        foot_slip_weight=foot_slip_weight,
        **overrides,
    )


def AS2WLocomotion(
    *, contact_observation: bool = False, sphere_wheels: bool = False, **overrides: Any
) -> EnvironmentConfig:
    # Each leg's wheel channel follows its position channels.
    actions: dict[str, ActionConfig] = {}
    for leg in ("FL", "FR", "RL", "RR"):
        actions[f"{leg}_position"] = JointPosition(
            joint_names=f"{leg}_(hip|thigh|calf)", scale=0.4
        )
        actions[f"{leg}_velocity"] = JointVelocity(joint_names=f"{leg}_foot", scale=10.0)
    return _velocity_locomotion(
        assetx_as2w_sphere if sphere_wheels else assetx_as2w,
        actions=actions,
        base="base_link", feet=r"(FL|FR|RL|RR)_foot",
        contact_observation=contact_observation, foot_slip_weight=None, **overrides,
    )


def M20SLocomotion(*, contact_observation: bool  =  False, **overrides: Any) -> EnvironmentConfig:
    # Each leg's wheel channel follows its position channels.
    actions: dict[str, ActionConfig] = {}
    for leg in ("fl", "fr", "hl", "hr"):
        actions[f"{leg}_position"] = JointPosition(
            joint_names=f"{leg}_(hipx|hipy|knee)_joint", scale=0.4
        )
        actions[f"{leg}_velocity"] = JointVelocity(
            joint_names=f"{leg}_wheel_joint", scale=10.0
        )
    return _velocity_locomotion(
        assetx_m20s,
        actions=actions,
        base="base_link", feet=r"(fl|fr|hl|hr)_wheel",
        contact_observation=contact_observation, foot_slip_weight=None, **overrides,
    )


def _velocity_locomotion(
    robot: Callable[[], RobotConfig],
    *,
    actions: dict[str, ActionConfig],
    base: str,
    feet: str,
    contact_observation: bool,
    foot_slip_weight: float | None,
    **overrides: Any,
) -> EnvironmentConfig:
    mdp = locomotion_mdp(VelocityMDP, base)
    mdp.actions = actions
    mdp.observations["command"] = ObservationGroup(
        {"velocity_command": velocity_command}
    )
    mdp.observations["privileged"] = ObservationGroup(
        {
            **dict(mdp.observations["privileged"].terms),
            "feet_lin_vel_w": BodyLinearVelocity("robot", feet),
            **(
                {"feet_loaded": ContactObservation("feet_ground")}
                if contact_observation
                else {}
            ),
        }
    )
    mdp.rewards = {
        **mdp.rewards,
        "tracking": RewardGroup(
            {
                "track_xy_velocity": TrackVelocity(
                    BodyLinearVelocity("robot", base, frame="yaw"),
                    slice(0, 2),
                    weight=1.5,
                    sigma=0.5,
                ),
                "track_yaw_rate": TrackVelocity(
                    BodyAngularVelocity("robot", base, frame="yaw"),
                    slice(2, 3),
                    weight=1.0,
                    sigma=0.5,
                ),
            }
        ),
        "regularization": RewardGroup(
            {
                "root_angvel_xy": BodyAngularVelocity2(
                    entity="robot",
                    body_names=base,
                    axis=(1.0, 1.0, 0.0),
                    frame="body",
                    weight=0.02,
                ),
                "root_linvel_z": BodyLinearVelocity2(
                    entity="robot",
                    body_names=base,
                    axis=(0.0, 0.0, 1.0),
                    frame="body",
                    weight=2.0,
                ),
                "action_rate": ActionRate(weight=0.01),
                "joint_energy1": JointEnergy1(weight=0.0002),
                # "joint_acc2": JointAcc2(weight=0.00001),
                "joint_vel2": JointVel2(weight=0.002),
                **(
                    {
                        "feet_air_time": FeetAirTime(sensor="feet_ground", weight=0.2),
                        "foot_slip": FootSlip(sensor="feet_ground", weight=foot_slip_weight),
                    }
                    if foot_slip_weight is not None
                    else {}
                ),
            }
        ),
    }
    mdp.terminations = {
        "tilt": TiltTermination(ProjectedGravity("robot", base)),
    }
    sensors = {
        "feet_ground": ContactSensorConfig("robot", feet, track_air_time=True)
    }
    defaults = {
        "robots": {"robot": robot},
        "mdp": mdp,
        "sensors": sensors | overrides.pop("sensors", {}),
    }
    return EnvironmentConfig(**(defaults | overrides))


from newrl.utils.config import cs

cs.store(
    "g1",
    DomainRandomizationConfig(
        mass=BodyMassScale(body_names="pelvis", bounds=(0.85, 1.15)),
        friction=ShapeFriction(
            body_names=r"(left|right)_ankle_roll_link", bounds=(1.0, 1.4)
        ),
        gains=JointGainScale(joint_names=".*", bounds=(0.9, 1.1)),
    ),
    group="randomization",
)

cs.store("wall", BorderConfig(), group="terrain.border")
cs.store("apron", BorderConfig(kind="apron", width=5.0), group="terrain.border")
cs.store("plane", PlaneTerrainConfig(), group="terrain")
cs.store("recon", ReconstructedTerrainConfig, group="terrain")
cs.store(
    "flat_tiles",
    GeneratedTerrainConfig(tiles={"flat": FlatTileConfig()}),
    group="terrain",
)
cs.store(
    "rough_easy",
    GeneratedTerrainConfig(
        tiles={
            "fine": RoughHeightfieldTileConfig(amplitude=0.04, noise_spacing=0.30),
            "coarse": RoughHeightfieldTileConfig(amplitude=0.06, noise_spacing=0.55),
            "slope": PyramidSlopeTileConfig(slope=0.25),
            "slope_inverted": PyramidSlopeTileConfig(slope=0.25, inverted=True),
        }
    ),
    group="terrain",
)
cs.store(
    "rough_hard",
    GeneratedTerrainConfig(
        tiles={
            "rough": RoughHeightfieldTileConfig(amplitude=0.10, noise_spacing=0.40),
            "stairs": StairsTileConfig(step_height=0.12, step_width=0.40),
            "slope": PyramidSlopeTileConfig(slope=0.25),
            "slope_inverted": PyramidSlopeTileConfig(slope=0.25, inverted=True),
        }
    ),
    group="terrain",
)
cs.store(
    "stepping_stones",
    GeneratedTerrainConfig(
        tiles={
            "stones": SteppingStonesTileConfig(),
            "stones_tall": SteppingStonesTileConfig(
                stone_size=0.45, gap=0.28, height_range=(0.05, 0.20)
            ),
        }
    ),
    group="terrain",
)
cs.store(
    "random_grid",
    GeneratedTerrainConfig(
        tiles={
            "grid_easy": RandomGridTileConfig(height_range=(0.0, 0.10)),
            "grid_hard": RandomGridTileConfig(cell_width=0.65, height_range=(0.0, 0.20)),
        }
    ),
    group="terrain",
)

for name, factory in (
    ("AnymalLocoFlat", AnymalLocomotion),
    ("SpotLocoFlat", SpotLocomotion),
    ("AS2LocoFlat", AS2Locomotion),
    ("AS2WLocoFlat", AS2WLocomotion),
    ("M20SLocoFlat", M20SLocomotion),
    ("G1LocoFlat", G1Locomotion),
    ("G1LAFAN1Tracking", G1MotionTracking),
):
    cs.store(name, factory(name=name, num_envs=4096), group="task")

cs.store(
    "AS2WSphereLocoFlat",
    AS2WLocomotion(name="AS2WSphereLocoFlat", num_envs=4096, sphere_wheels=True),
    group="task",
)

cs.store("G1Tabletop", G1Tabletop(), group="task")
