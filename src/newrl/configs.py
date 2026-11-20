"""Scene, simulation, and task configuration for the batched environment."""

from collections.abc import Callable
from dataclasses import dataclass, field

import newton

from .mdp.base import MDP
from .mdp.episode_ends import Fall, Timeout
from .mdp.observations import (
    BodyAngularVelocity,
    BodyHeight,
    BodyLinearVelocity,
    EntityField,
    ProjectedGravity,
    previous_action,
    velocity_command,
)
from .mdp.rewards import ActionRate, TrackVelocity
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
    policy_observation_keys: tuple[str, ...] = ()
    critic_observation_keys: tuple[str, ...] = ()


def locomotion_mdp(mdp_type: type[MDP] = MDP) -> MDP:
    return mdp_type(
        observations={
            "root_lin_vel_b": BodyLinearVelocity("robot", "base", frame="body"),
            "root_ang_vel_b": BodyAngularVelocity("robot", "base", frame="body"),
            "projected_gravity": ProjectedGravity("robot", "base"),
            "root_height": BodyHeight("robot", "base"),
            "joint_pos": EntityField("robot", "joint_pos"),
            "joint_vel": EntityField("robot", "joint_vel"),
            "previous_action": previous_action,
        },
        truncations={"timeout": Timeout(1000)},
    )


_LOCOMOTION_OBSERVATIONS = (
    "root_lin_vel_b",
    "root_ang_vel_b",
    "projected_gravity",
    "root_height",
    "joint_pos",
    "joint_vel",
    "previous_action",
)


def AnymalLocomotion(**overrides) -> EnvironmentConfig:
    mdp = locomotion_mdp(VelocityMDP)
    mdp.observations["velocity_command"] = velocity_command
    mdp.observations["feet_lin_vel_w"] = BodyLinearVelocity(
        "robot", r"(LF|LH|RF|RH)_FOOT"
    )
    mdp.rewards = {
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
        "action_rate": ActionRate(weight=0.01),
    }
    mdp.terminations = {
        "fall": Fall(BodyHeight("robot", "base"), ProjectedGravity("robot", "base"))
    }
    observation_keys = (*_LOCOMOTION_OBSERVATIONS, "feet_lin_vel_w", "velocity_command")
    defaults = {
        "robots": {"robot": newton_anymal_c},
        "mdp": mdp,
        "policy_observation_keys": observation_keys,
        "critic_observation_keys": observation_keys,
    }
    return EnvironmentConfig(**(defaults | overrides))


def SpotLocomotion(**overrides) -> EnvironmentConfig:
    defaults = {
        "robots": {"robot": assetx_spot},
        "mdp": locomotion_mdp(),
        "policy_observation_keys": _LOCOMOTION_OBSERVATIONS,
        "critic_observation_keys": _LOCOMOTION_OBSERVATIONS,
    }
    return EnvironmentConfig(**(defaults | overrides))


def AS2Locomotion(**overrides) -> EnvironmentConfig:
    defaults = {
        "robots": {"robot": assetx_as2},
        "mdp": locomotion_mdp(),
        "policy_observation_keys": _LOCOMOTION_OBSERVATIONS,
        "critic_observation_keys": _LOCOMOTION_OBSERVATIONS,
    }
    return EnvironmentConfig(**(defaults | overrides))
