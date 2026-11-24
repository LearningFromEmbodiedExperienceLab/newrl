"""Scene, simulation, and task configuration for the batched environment."""

from collections.abc import Callable
from dataclasses import dataclass, field

import newton

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
from .robots import (
    RobotConfig, assetx_as2, assetx_as2w, assetx_as2w_sphere,
    assetx_m20s, assetx_spot, newton_anymal_c,
    assetx_g1,
)
from .sensors import SensorConfig, TerrainContactConfig
from .solver_adapters import MujocoSimConfig, SteelSimConfig
from .terrain import (
    BorderConfig,
    FlatTileConfig,
    GeneratedTerrainConfig,
    PlaneTerrainConfig,
    PyramidSlopeTileConfig,
    ReconstructedTerrainConfig,
    RoughHeightfieldTileConfig,
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
    robots: dict[str, Callable[[], RobotConfig]] = field(default_factory=dict)
    mdp: MDP = field(default_factory=MDP)
    physical_fields: tuple[tuple[str, str], ...] = ()
    output_physical_fields: tuple[tuple[str, str], ...] = ()
    output_mdp_fields: tuple[str, ...] = ()
    name: str = "custom"
    collision: str = "newton"
    sensors: dict[str, SensorConfig] = field(default_factory=dict)

    def __post_init__(self):
        if self.collision not in ("newton", "mujoco", "newrl"):
            raise ValueError("collision must be newton, mujoco, or newrl")


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
        rewards={"survival": RewardGroup({"survival": Survival(weight=1.0)})},
        truncations={"timeout": Timeout(1000)},
    )


def G1MotionTracking(**overrides) -> EnvironmentConfig:
    """50 Hz G1 tracking of the official retargeted LAFAN1 library."""
    bodies = (
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    )
    mdp = locomotion_mdp(MotionTrackingMDP, base="pelvis")
    mdp.observations["command"] = ObservationGroup({"reference": ReferenceMotion()})
    mdp.rewards["tracking"] = RewardGroup({
        "body_pos": TrackBodyMotion(1.0, 0.3, "position", body_names=bodies, anchor_body="pelvis"),
        "body_ori": TrackBodyMotion(1.0, 0.4, "orientation", body_names=bodies, anchor_body="pelvis"),
        "body_lin_vel": TrackBodyMotion(1.0, 1.0, "linear_velocity", body_names=bodies),
        "body_ang_vel": TrackBodyMotion(1.0, 3.14, "angular_velocity", body_names=bodies),
        "root_pos": TrackBodyMotion(0.5, 0.3, body_names="pelvis"),
        "root_ori": TrackBodyMotion(0.5, 0.4, "orientation", body_names="pelvis"),
        "action_rate": ActionRate(0.05),
        "joint_limit": JointLimit(2.0),
    })
    mdp.terminations = {
        "height": TrackingDeviation(BodyHeight("robot", "pelvis")),
        "tilt": TrackingDeviation(BodyHeight("robot", "pelvis"), "tilt", 0.8),
    }
    mdp.truncations["clip_end"] = ClipEnd()
    mdp.truncations["timeout"] = Timeout(1000)
    options = dict(num_envs=4096, robots={"robot": assetx_g1}, mdp=mdp, name="G1LAFAN1Tracking")
    options.update(overrides)
    return EnvironmentConfig(**options)


def AnymalLocomotion(
    *,
    contact_observation: bool = False,
    foot_slip_weight: float = 0.1,
    **overrides,
) -> EnvironmentConfig:
    return _quadruped_locomotion(
        newton_anymal_c,
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
    **overrides,
) -> EnvironmentConfig:
    return _quadruped_locomotion(
        assetx_spot,
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
    **overrides,
) -> EnvironmentConfig:
    return _quadruped_locomotion(
        assetx_as2,
        base="base_link",
        feet=".*_foot",
        contact_observation=contact_observation,
        foot_slip_weight=foot_slip_weight,
        **overrides,
    )


def AS2WLocomotion(
    *, contact_observation: bool = False, sphere_wheels: bool = False, **overrides
) -> EnvironmentConfig:
    config = _quadruped_locomotion(
        assetx_as2w_sphere if sphere_wheels else assetx_as2w,
        base="base_link", feet=r"(FL|FR|RL|RR)_foot",
        contact_observation=contact_observation, foot_slip_weight=0.0, **overrides,
    )
    del config.mdp.rewards["regularization"].terms["foot_slip"]
    del config.mdp.rewards["regularization"].terms["feet_air_time"]
    return config


def M20SLocomotion(*, contact_observation: bool = False, **overrides) -> EnvironmentConfig:
    config = _quadruped_locomotion(
        assetx_m20s, base="base_link", feet=r"(fl|fr|hl|hr)_wheel",
        contact_observation=contact_observation, foot_slip_weight=0.0, **overrides,
    )
    del config.mdp.rewards["regularization"].terms["foot_slip"]
    del config.mdp.rewards["regularization"].terms["feet_air_time"]
    return config


def _quadruped_locomotion(
    robot: Callable[[], RobotConfig],
    *,
    base: str,
    feet: str,
    contact_observation: bool,
    foot_slip_weight: float,
    **overrides,
) -> EnvironmentConfig:
    mdp = locomotion_mdp(VelocityMDP, base)
    mdp.observations["command"] = ObservationGroup(
        {"velocity_command": velocity_command}
    )
    mdp.observations["privileged"] = ObservationGroup(
        {
            **mdp.observations["privileged"].terms,
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
                "feet_air_time": FeetAirTime(sensor="feet_ground", weight=0.2),
                "foot_slip": FootSlip(sensor="feet_ground", weight=foot_slip_weight),
            }
        ),
    }
    mdp.terminations = {
        "tilt": TiltTermination(ProjectedGravity("robot", base)),
    }
    sensors = {
        "feet_ground": TerrainContactConfig("robot", feet, track_air_time=True)
    }
    defaults = {
        "robots": {"robot": robot},
        "mdp": mdp,
        "sensors": sensors | overrides.pop("sensors", {}),
    }
    return EnvironmentConfig(**(defaults | overrides))


from newrl.utils.config import cs

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

for name, factory in (
    ("AnymalLocoFlat", AnymalLocomotion),
    ("SpotLocoFlat", SpotLocomotion),
    ("AS2LocoFlat", AS2Locomotion),
    ("AS2WLocoFlat", AS2WLocomotion),
    ("M20SLocoFlat", M20SLocomotion),
    ("G1LAFAN1Tracking", G1MotionTracking),
):
    cs.store(name, factory(name=name, num_envs=4096), group="task")

cs.store(
    "AS2WSphereLocoFlat",
    AS2WLocomotion(name="AS2WSphereLocoFlat", num_envs=4096, sphere_wheels=True),
    group="task",
)
