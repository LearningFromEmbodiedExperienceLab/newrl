"""Train a clipped PPO policy on a NewRL environment."""

from dataclasses import dataclass, field

import torch

from newrl.algos.ppo import PPO, PPOConfig
from newrl.algos.rollout import collect_rollout
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.utils.config import main
from newrl.utils.logging import RunLogger, WandbConfig


@dataclass
class TrainConfig:
    task: EnvironmentConfig
    algo: PPOConfig
    seed: int = 0
    iterations: int = 100
    logger: WandbConfig = field(default_factory=WandbConfig)


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
    logger = RunLogger(cfg.logger, cfg)
    try:
        for iteration in range(cfg.iterations):
            batch, carry = collect_rollout(env, policy, cfg.algo.update_every, carry)
            metrics = policy.train_op(batch)
            frames = (iteration + 1) * env.num_envs * cfg.algo.update_every
            logger.log({**metrics, "env_frames": frames}, step=iteration)
            summary = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"iter={iteration} env_frames={frames} {summary}", flush=True)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
