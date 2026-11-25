"""CPU checks for the GenPO++ flow and KL adaptation."""

import unittest

import torch
from tensordict import TensorDict
from torch import nn

from newrl.algos.genpo import GenPO, GenPOConfig, ReversibleFlowActor


class GenPOTests(unittest.TestCase):
    def test_flow_inversion_and_behavior_log_prob(self):
        torch.manual_seed(0)
        cfg = GenPOConfig(actor_hidden=(8, 8))
        actor = ReversibleFlowActor(3, 2, nn.Mish, cfg)
        observation = torch.randn(4, 3)
        noise = torch.randn(4, 4)
        action, history = actor.integrate(observation, *noise.chunk(2, dim=-1))
        torch.testing.assert_close(actor.invert(observation, action, history), noise)

        action, history, log_prob = actor.sample(observation)
        self.assertEqual(log_prob.shape, (4, 1))
        ratio = (actor.log_prob(observation, action, history) - log_prob).exp()
        torch.testing.assert_close(ratio, torch.ones_like(log_prob))
        action, history, _ = actor.sample(observation, deterministic=True)
        torch.testing.assert_close(
            actor.invert(observation, action, history), torch.zeros_like(noise)
        )

    def test_kl_adapts_all_actor_groups(self):
        cfg = GenPOConfig(
            muon=True,
            actor_hidden=(8, 8),
            critic_hidden=(8, 8),
            num_epochs=1,
            num_minibatches=1,
        )
        policy = GenPO(cfg, policy_dim=3, critic_dim=3, action_dim=2)
        groups = policy.optimizer.param_groups
        self.assertEqual(len(groups), 4)
        for log_ratio, actor_lr in (
            (-0.2, cfg.lr / 1.5),
            (-0.01, cfg.lr),
            (0.1, cfg.lr),
        ):
            obs = torch.randn(2, 2, 3)
            with torch.no_grad():
                action, history, log_prob = policy.actor.sample(obs)
            batch = TensorDict(
                {
                    "policy_observation": obs,
                    "critic_observation": obs,
                    ("next", "critic_observation"): obs,
                    "action": action,
                    "flow_hist": history,
                    "action_log_prob": log_prob - log_ratio,
                    ("next", "reward"): {"tracking": torch.randn(2, 2, 1)},
                    ("next", "done"): torch.zeros(2, 2, 1, dtype=torch.bool),
                    ("next", "terminated"): torch.zeros(2, 2, 1, dtype=torch.bool),
                },
                batch_size=[2, 2],
            )
            metrics = policy.train_op(batch)
            self.assertAlmostEqual(metrics["actor/aug_kl"], -log_ratio, places=5)
            for group, expected in zip(
                groups, (actor_lr, cfg.lr, actor_lr, cfg.lr), strict=True
            ):
                self.assertAlmostEqual(group["lr"], expected)
