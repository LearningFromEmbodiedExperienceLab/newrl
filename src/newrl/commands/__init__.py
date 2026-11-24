"""Built-in NewRL commands."""

from typing import TYPE_CHECKING
"""Shared environment configuration for the training and playback commands."""

from dataclasses import replace

if TYPE_CHECKING:
    from newrl.algos.interfaces import LearnerConfig
    from newrl.configs import EnvironmentConfig


def with_used_observations(
    task: "EnvironmentConfig",
    algo: "LearnerConfig | None",
) -> "EnvironmentConfig":
    """Select learner inputs on a copy of the task so checkpoints keep all groups."""
    default_keys = () if algo is None else None
    policy_keys = getattr(algo, "policy_obs_keys", default_keys)
    critic_keys = getattr(algo, "critic_obs_keys", default_keys)
    if policy_keys is None or critic_keys is None:
        return task
    used = set(policy_keys) | set(critic_keys)
    observations = {
        name: group for name, group in task.mdp.observations.items() if name in used
    }
    return replace(task, mdp=replace(task.mdp, observations=observations))
