"""Train a clipped PPO policy on a NewRL environment."""

import torch
from dataclasses import dataclass
from newrl.algos.ppo import PPO
from newrl.algos.rollout import collect_rollout
from newrl.env import StatefulEnvironment
from newrl.utils.config import main

# Imported after ConfigStore exists so task and algo modules can register presets.
from newrl.algos.ppo import PPOConfig  # noqa: E402
from newrl.configs import EnvironmentConfig  # noqa: E402


@dataclass
class TrainConfig:
    task: EnvironmentConfig
    algo: PPOConfig
    seed: int = 0
    iterations: int = 100


def _group_width(mdp, groups: tuple[str, ...]) -> int:
    return sum(mdp.observations[name].shape.numel() for name in groups)


@main(TrainConfig)
def main(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    cfg.task.mdp.seed = cfg.seed
    env = StatefulEnvironment(cfg.task)
    policy = PPO(
        cfg.algo,
        policy_dim=_group_width(env.mdp, cfg.algo.policy_obs_keys),
        critic_dim=_group_width(env.mdp, cfg.algo.critic_obs_keys),
        action_dim=env.action_dim,
    ).to(env.device)
    carry = env.reset()

    for iteration in range(cfg.iterations):
        batch, carry = collect_rollout(env, policy, cfg.algo.update_every, carry)
        metrics = policy.train_op(batch)
        summary = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"iter={iteration} {summary}", flush=True)


if __name__ == "__main__":
    main()
