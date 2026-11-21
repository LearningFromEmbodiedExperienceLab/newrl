"""Train a clipped PPO policy on a NewRL environment."""

import argparse

import torch

from newrl.algos.ppo import PPO, PPOConfig
from newrl.algos.rollout import collect_rollout
from newrl.configs import AnymalLocomotion
from newrl.env import StatefulEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--update-every", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.update_every, args.epochs, args.minibatches) < 1:
        parser.error("training sizes must be positive")

    torch.manual_seed(args.seed)
    config = AnymalLocomotion(num_envs=args.num_envs, device=args.device)
    config.mdp.seed = args.seed
    env = StatefulEnvironment(config)
    # TODO: statically obtain observation dimensions
    probe = env.reset()
    policy = PPO(
        PPOConfig(
            update_every=args.update_every,
            num_epochs=args.epochs,
            num_minibatches=args.minibatches,
            lr=args.lr,
        ),
        policy_dim=probe["policy_observation"].shape[-1],
        critic_dim=probe["critic_observation"].shape[-1],
        action_dim=env.action_dim,
    ).to(env.device)
    # TODO: carry should be a tensordict for flexibility
    carry = (probe["policy_observation"], probe["critic_observation"])

    for iteration in range(args.iterations):
        batch, carry = collect_rollout(env, policy, args.update_every, carry)
        metrics = policy.train_op(batch)
        summary = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"iter={iteration} {summary}", flush=True)


if __name__ == "__main__":
    main()
