"""Rollout fields consumed by on-policy updates.

Time is dimension 1, so a collected batch has shape ``[num_envs, steps, ...]``.
``policy_observation`` is the observation that produced the action.
``critic_observation`` is the value input at that same decision.
``("next", "critic_observation")`` is the critic observation of the true successor,
including on termination or truncation, and it is recorded before any reset.
``("next", "reward")`` holds a tensor per group with one column per term.
``("next", "terminated")`` drops the value bootstrap. ``("next", "done")`` is termination or truncation
and stops the advantage trace at the episode boundary.
"""

from typing import TYPE_CHECKING

from collections.abc import Iterator

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from newrl.mdp.base import MDP

POLICY_OBSERVATION = "policy_observation"
CRITIC_OBSERVATION = "critic_observation"
NEXT_CRITIC_OBSERVATION = ("next", CRITIC_OBSERVATION)
ACTION = "action"
ACTION_LOG_PROB = "action_log_prob"
REWARD = ("next", "reward")
TERMINATED = ("next", "terminated")
DONE = ("next", "done")

BATCH_KEYS = (
    POLICY_OBSERVATION,
    CRITIC_OBSERVATION,
    NEXT_CRITIC_OBSERVATION,
    ACTION,
    ACTION_LOG_PROB,
    REWARD,
    TERMINATED,
    DONE,
)


def group_width(mdp: "MDP", groups: tuple[str, ...]) -> int:
    return sum(mdp.observations[name].shape.numel() for name in groups)


def stack_groups(observation: TensorDict, groups: tuple[str, ...]) -> torch.Tensor:
    """Concatenate observation groups in the given order."""
    missing = [name for name in groups if name not in observation]
    if missing:
        raise KeyError(f"Observation is missing groups {missing}")
    return torch.cat([observation[name] for name in groups], dim=-1)


def aggregate_reward(batch: TensorDict) -> torch.Tensor:
    groups = batch[REWARD]
    scalar_shape = (*batch.batch_size, 1)
    if isinstance(groups, torch.Tensor):
        if groups.shape != scalar_shape:
            raise ValueError(
                f"Expected reward shape {scalar_shape}, got {tuple(groups.shape)}"
            )
        return groups
    parts = []
    for name, values in groups.items():
        if values.ndim != batch.ndim + 1 or values.shape[:-1] != batch.batch_size:
            raise ValueError(f"Expected one reward feature dimension for group {name!r}")
        parts.append(values)
    if not parts:
        return batch[DONE].new_zeros(scalar_shape, dtype=torch.float32)
    return torch.cat(parts, dim=-1).sum(dim=-1, keepdim=True)


def minibatches(batch: TensorDict, num_minibatches: int) -> Iterator[TensorDict]:
    flat = batch.reshape(-1)
    usable = (flat.shape[0] // num_minibatches) * num_minibatches
    if usable == 0:
        raise ValueError(
            f"Batch of {flat.shape[0]} transitions is smaller than num_minibatches={num_minibatches}"
        )
    permutation = torch.randperm(usable, device=flat.device).reshape(
        num_minibatches, -1
    )
    for index in permutation:
        yield flat[index]


def collect_metrics(
    minibatch_metrics: list[dict[str, torch.Tensor]],
    rollout_metrics: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Average the last epoch and transfer all tensor metrics together."""
    keys = tuple(minibatch_metrics[0])
    rows = [torch.stack([info[key] for key in keys]) for info in minibatch_metrics]
    epoch_mean = torch.stack(rows).mean(dim=0)
    values = torch.cat((epoch_mean, torch.stack(tuple(rollout_metrics.values()))))
    return dict(zip((*keys, *rollout_metrics), values.cpu().tolist(), strict=True))
