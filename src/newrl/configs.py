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
    EntityField,
    ProjectedGravity,
    previous_action,
    velocity_command,
)
from .mdp.rewards import ActionRate, JointAcc2, JointEnergy1, JointVel2, TrackVelocity
from .mdp.velocity import VelocityMDP
from .robots import RobotConfig, assetx_as2, assetx_spot, newton_anymal_c
from .terrain import PlaneTerrainConfig, TerrainConfig


@dataclass
class EnvironmentConfig:
    num_envs: int = 1
    device: str = "cuda:0"
    sim_dt: float = 0.005
    substeps: int = 4
    jacobian: str = "sparse"
    solver: str = "newton"
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 100.0
    iterations: int = 20
    ls_iterations: int = 40
    nconmax: int = 80
    njmax: int = 100
    ground_stiffness: float = 1.0e3
    ground_damping: float = 1.0e2
    graph_capture: bool = False
    deterministic: bool = False
    terrain: TerrainConfig | str | Callable[[newton.ModelBuilder], None] = field(
        default_factory=PlaneTerrainConfig
    )
    robots: dict[str, Callable[[], RobotConfig]] = field(default_factory=dict)
    mdp: MDP = field(default_factory=MDP)
    physical_fields: tuple[tuple[str, str], ...] = ()


def locomotion_mdp(mdp_type: type[MDP] = MDP) -> MDP:
    return mdp_type(
        observations={
            "proprio": ObservationGroup(
                {
                    "joint_pos": EntityField("robot", "joint_pos"),
                    "joint_vel": EntityField("robot", "joint_vel"),
                    "projected_gravity": ProjectedGravity("robot", "base"),
                    "previous_action": previous_action,
                    "root_ang_vel_b": BodyAngularVelocity(
                        "robot", "base", frame="body"
                    ),
                }
            ),
            "privileged": ObservationGroup(
                {
                    "root_lin_vel_b": BodyLinearVelocity("robot", "base", frame="body"),
                    "root_height": BodyHeight("robot", "base"),
                }
            ),
        },
        truncations={"timeout": Timeout(1000)},
    )


def AnymalLocomotion(**overrides) -> EnvironmentConfig:
    mdp = locomotion_mdp(VelocityMDP)
    mdp.observations["command"] = ObservationGroup(
        {"velocity_command": velocity_command}
    )
    mdp.observations["privileged"] = ObservationGroup(
        {
            **mdp.observations["privileged"].terms,
            "feet_lin_vel_w": BodyLinearVelocity("robot", r"(LF|LH|RF|RH)_FOOT"),
        }
    )
    mdp.rewards = {
        "tracking": RewardGroup(
            {
                "track_xy_velocity": TrackVelocity(
                    BodyLinearVelocity("robot", "base", frame="yaw"),
                    slice(0, 2),
                    weight=1.0,
                    sigma=0.5,
                ),
                "track_yaw_rate": TrackVelocity(
                    BodyAngularVelocity("robot", "base", frame="yaw"),
                    slice(2, 3),
                    weight=0.5,
                    sigma=0.5,
                ),
            }
        ),
        "regularization": RewardGroup(
            {
                "action_rate": ActionRate(weight=0.01),
                "joint_energy1": JointEnergy1(weight=0.01),
                "joint_acc2": JointAcc2(weight=0.00001),
                "joint_vel2": JointVel2(weight=0.01),
            }
        ),
    }
    mdp.terminations = {
        "height": HeightTermination(BodyHeight("robot", "base")),
        "tilt": TiltTermination(ProjectedGravity("robot", "base")),
    }
    defaults = {"robots": {"robot": newton_anymal_c}, "mdp": mdp}
    return EnvironmentConfig(**(defaults | overrides))


def SpotLocomotion(**overrides) -> EnvironmentConfig:
    defaults = {"robots": {"robot": assetx_spot}, "mdp": locomotion_mdp()}
    return EnvironmentConfig(**(defaults | overrides))


def AS2Locomotion(**overrides) -> EnvironmentConfig:
    defaults = {"robots": {"robot": assetx_as2}, "mdp": locomotion_mdp()}
    return EnvironmentConfig(**(defaults | overrides))
