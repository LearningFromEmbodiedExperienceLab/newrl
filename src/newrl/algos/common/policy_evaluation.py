"""GAE over a fixed rollout window.

Inputs have shape ``[N, T, 1]``. Reward scaling stays with the algorithm.
"""

import torch


def generalized_advantage(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    done: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return advantages and returns for one rollout window.

    ``next_value`` is ``V`` at the true successor. ``terminated`` removes that
    bootstrap. ``done`` stops the trace from crossing into the next episode, so
    a truncation still bootstraps while a later reset does not leak backward.
    """
    reward = reward.float()
    value = value.float()
    next_value = next_value.float()
    if reward.shape != value.shape or reward.shape != next_value.shape:
        raise ValueError("reward, value, and next_value must share a shape")
    if reward.ndim != 3 or reward.shape[-1] != 1:
        raise ValueError(f"Expected shape [N, T, 1], got {tuple(reward.shape)}")
    terminated = terminated.to(dtype=torch.bool, device=reward.device)
    done = done.to(dtype=torch.bool, device=reward.device)
    if terminated.shape != reward.shape or done.shape != reward.shape:
        raise ValueError("terminated and done must match the reward shape")

    advantages = torch.zeros_like(reward)
    nonterminal = (~terminated).to(reward.dtype)
    not_done = (~done).to(reward.dtype)
    discount = reward.new_tensor(gamma)
    trace_decay = reward.new_tensor(lam)
    advantage = torch.zeros_like(reward[:, 0])
    for step in reversed(range(reward.shape[1])):
        bootstrap = next_value[:, step] * nonterminal[:, step]
        delta = reward[:, step] + discount * bootstrap - value[:, step]
        advantage = delta + discount * trace_decay * not_done[:, step] * advantage
        advantages[:, step] = advantage
    return advantages, advantages + value
