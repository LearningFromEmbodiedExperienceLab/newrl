"""CPU checks for GAE, normalization, the PPO update, and rollout assembly."""

import unittest

import torch
from tensordict import TensorDict

from newrl.algos.common.modules import NormalActor
from newrl.algos.common.normalization import EmpiricalNormalization, ReturnNormalization
from newrl.algos.common.policy_evaluation import generalized_advantage
from newrl.algos.ppo import PPO, PPOConfig
from newrl.algos.rollout import collect_rollout


class AdvantageTests(unittest.TestCase):
    def test_truncation_bootstraps_and_termination_does_not(self):
        reward = torch.tensor([[[1.0], [1.0]]])
        value = torch.zeros_like(reward)
        next_value = torch.tensor([[[0.0], [10.0]]])
        continuing = torch.zeros_like(reward, dtype=torch.bool)
        truncated = torch.tensor([[[False], [True]]])
        terminated = truncated.clone()

        truncated_adv, _ = generalized_advantage(
            reward, continuing, truncated, value, next_value, gamma=0.5, lam=1.0
        )
        terminated_adv, _ = generalized_advantage(
            reward, terminated, truncated, value, next_value, gamma=0.5, lam=1.0
        )

        self.assertTrue(torch.allclose(truncated_adv, torch.tensor([[[4.0], [6.0]]])))
        self.assertTrue(torch.allclose(terminated_adv, torch.tensor([[[1.5], [1.0]]])))

    def test_done_stops_the_trace_at_an_episode_boundary(self):
        reward = torch.tensor([[[1.0], [10.0]]])
        value = torch.zeros_like(reward)
        next_value = torch.zeros_like(reward)
        terminated = torch.zeros_like(reward, dtype=torch.bool)
        done = torch.tensor([[[True], [False]]])
        advantage, returns = generalized_advantage(
            reward, terminated, done, value, next_value, gamma=0.5, lam=1.0
        )
        self.assertTrue(torch.allclose(advantage, torch.tensor([[[1.0], [10.0]]])))
        self.assertTrue(torch.allclose(returns, advantage))


class NormalizationTests(unittest.TestCase):
    def test_freeze_keeps_observation_statistics_fixed(self):
        norm = EmpiricalNormalization(2, decay=1.0)
        first = torch.zeros(4, 2)
        second = torch.ones(4, 2)
        norm(first)
        mean, std = norm.mean_std()
        with norm.freeze():
            frozen = norm(second)
            self.assertEqual(float(norm.count), 4.0)
        self.assertTrue(torch.allclose(frozen, (second - mean) / std))
        norm(second)
        self.assertEqual(float(norm.count), 8.0)

    def test_return_scale_resets_after_done(self):
        norm = ReturnNormalization(gamma=1.0)
        reward = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]]])
        done = torch.tensor([[[True], [False]], [[False], [False]]])
        norm.update(reward, done)
        self.assertEqual(float(norm.returns[0]), 1.0)
        self.assertEqual(float(norm.returns[1]), 2.0)


class RolloutTests(unittest.TestCase):
    def test_successor_is_kept_when_a_world_resets(self):
        env = _ScriptedEnvironment()
        batch, carry = collect_rollout(env, _ZeroPolicy(), steps=2)
        self.assertEqual(tuple(batch.batch_size), (2, 2))
        self.assertTrue(
            torch.equal(batch["policy_observation"][:, 0], torch.tensor([[0.0], [0.0]]))
        )
        self.assertTrue(
            torch.equal(batch["next_critic_observation"][:, 0], torch.tensor([[1.0], [1.0]]))
        )
        self.assertTrue(
            torch.equal(batch["policy_observation"][:, 1], torch.tensor([[50.0], [1.0]]))
        )
        self.assertTrue(torch.equal(carry[0], torch.tensor([[51.0], [2.0]])))


class PPOTests(unittest.TestCase):
    def test_reward_scale_and_update(self):
        torch.manual_seed(0)
        policy = PPO(
            PPOConfig(
                num_epochs=2,
                num_minibatches=2,
                actor_hidden=(8,),
                critic_hidden=(8,),
                gamma=0.5,
                lam=1.0,
                lr=1e-2,
            ),
            policy_dim=3,
            critic_dim=3,
            action_dim=2,
        )
        batch = _random_batch(num_envs=4, steps=4, policy_dim=3, action_dim=2)
        scaled = policy._scaled_reward(batch)
        self.assertTrue(torch.allclose(scaled, batch["reward"] * 0.5))
        policy.critic.head.weight.data.zero_()
        policy.critic.head.bias.data.zero_()
        with torch.no_grad():
            value = policy.critic(batch["critic_observation"])
            next_value = policy.critic(batch["next_critic_observation"])
        self.assertTrue(torch.allclose(value, torch.zeros_like(value)))
        advantage, _ = generalized_advantage(
            scaled,
            batch["terminated"],
            batch["done"],
            value,
            next_value,
            gamma=0.5,
            lam=1.0,
        )
        self.assertEqual(tuple(advantage.shape), (4, 4, 1))

        before = policy.policy_norm.count.clone()
        actor_before = policy.actor.mean.weight.detach().clone()
        metrics = policy.train_op(batch)
        self.assertTrue(torch.equal(policy.policy_norm.count, before))
        self.assertGreater(float(policy.critic_norm.count), 0.0)
        self.assertFalse(torch.equal(policy.actor.mean.weight, actor_before))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertIn("actor/policy_loss", metrics)

    def test_std_stays_inside_its_bounds(self):
        actor = NormalActor(4, 3, hidden=(8,), min_std=0.05, max_std=0.4)
        scale = actor.scale()
        self.assertTrue(torch.all(scale > 0.05))
        self.assertTrue(torch.all(scale < 0.4))


class _ZeroPolicy:
    def act(self, observation):
        return torch.zeros(observation.shape[0], 2), torch.zeros(observation.shape[0], 1)


class _ScriptedEnvironment:
    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self._observation = torch.zeros(self.num_envs, 1)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._observation[:] = 0
        else:
            self._observation[env_ids] = 50
        return self._transition(torch.zeros(self.num_envs, 1), torch.zeros(self.num_envs, 1, dtype=torch.bool))

    def step(self, action):
        self._observation = self._observation + 1
        done = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        done[0, 0] = bool(self._observation[0, 0] == 1)
        return self._transition(torch.ones(self.num_envs, 1), done)

    def _transition(self, reward, done):
        return TensorDict(
            {
                "policy_observation": self._observation.clone(),
                "critic_observation": self._observation.clone(),
                "reward": reward,
                "terminated": torch.zeros_like(done),
                "done": done,
            },
            batch_size=[self.num_envs],
        )


def _random_batch(num_envs, steps, policy_dim, action_dim):
    observation = torch.randn(num_envs, steps, policy_dim)
    return TensorDict(
        {
            "policy_observation": observation,
            "critic_observation": observation.clone(),
            "next_critic_observation": torch.randn(num_envs, steps, policy_dim),
            "action": torch.randn(num_envs, steps, action_dim),
            "action_log_prob": torch.zeros(num_envs, steps, 1),
            "reward": torch.randn(num_envs, steps, 1),
            "terminated": torch.zeros(num_envs, steps, 1, dtype=torch.bool),
            "done": torch.zeros(num_envs, steps, 1, dtype=torch.bool),
        },
        batch_size=[num_envs, steps],
    )


if __name__ == "__main__":
    unittest.main()
