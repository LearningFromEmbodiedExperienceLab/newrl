"""CPU checks for GAE, normalization, the PPO update, and rollout assembly."""

import io
import unittest

import torch
from tensordict import TensorDict

from newrl.algos.common.modules import NormalActor
from newrl.algos.common.normalization import EmpiricalNormalization, ReturnNormalization
from newrl.algos.common.optimizer import MuonAdamW
from newrl.algos.common.policy_evaluation import generalized_advantage
from newrl.algos.ppo import PPO, PPOConfig, _aggregate_reward
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
        batch, carry = collect_rollout(env, _RecordingPolicy(), 2, env.reset())
        self.assertEqual(tuple(batch.batch_size), (2, 2))
        torch.testing.assert_close(batch["reward", "tracking"], torch.ones(2, 2, 1))
        self.assertTrue(
            torch.equal(batch["policy_observation"][:, 0], torch.tensor([[0.0], [0.0]]))
        )
        self.assertTrue(
            torch.equal(
                batch["next_critic_observation"][:, 0],
                torch.tensor([[1.0, 10.0], [1.0, 10.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["policy_observation"][:, 1], torch.tensor([[50.0], [1.0]])
            )
        )
        self.assertTrue(
            torch.equal(carry["observation", "state"], torch.tensor([[51.0], [2.0]]))
        )
        self.assertEqual(
            batch["mdp_state", "episode_step"].tolist(), [[[0], [0]], [[0], [1]]]
        )
        self.assertEqual(carry["mdp_state", "episode_step"].tolist(), [[1], [2]])


class PPOTests(unittest.TestCase):
    def test_training_checkpoint_restores_model_normalizers_and_optimizer(self):
        for muon in (False, True):
            with self.subTest(muon=muon):
                cfg = PPOConfig(
                    muon=muon,
                    normalize_reward=True,
                    actor_hidden=(8, 8),
                    critic_hidden=(8, 8),
                    num_epochs=1,
                    num_minibatches=2,
                )
                dims = dict(policy_dim=3, critic_dim=3, action_dim=2)
                policy = PPO(cfg, **dims)
                batch = _random_batch(num_envs=4, steps=4, policy_dim=3, action_dim=2)
                policy.policy_norm.update(batch["policy_observation"])
                policy.train_op(batch)
                checkpoint = io.BytesIO()
                torch.save(policy.training_state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = PPO(cfg, **dims)
                restored.load_training_state_dict(torch.load(checkpoint, weights_only=True))
                torch.testing.assert_close(restored.state_dict(), policy.state_dict())

                torch.manual_seed(0)
                expected = policy.train_op(batch)
                torch.manual_seed(0)
                actual = restored.train_op(batch)
                self.assertEqual(actual, expected)
                torch.testing.assert_close(restored.state_dict(), policy.state_dict())

    def test_muon_updates_hidden_matrices_and_restores_live_groups(self):
        cfg = PPOConfig(
            muon=True,
            actor_hidden=(8, 8),
            critic_hidden=(8, 8),
            num_epochs=1,
            num_minibatches=2,
            weight_decay=0.02,
        )
        policy = PPO(cfg, policy_dim=3, critic_dim=3, action_dim=2)
        optimizer = policy.optimizer
        self.assertIsInstance(optimizer, MuonAdamW)
        expected = [
            policy.actor.backbone.layers[3].weight,
            policy.critic.backbone.layers[3].weight,
        ]
        muon_params = [
            p for group in optimizer.muon.param_groups for p in group["params"]
        ]
        self.assertEqual([id(p) for p in muon_params], [id(p) for p in expected])
        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertCountEqual(map(id, all_params), map(id, policy.parameters()))
        self.assertTrue(
            all(group["weight_decay"] == cfg.weight_decay for group in optimizer.param_groups)
        )

        tracked = (*muon_params, policy.actor.mean.weight)
        before = [p.detach().clone() for p in tracked]
        batch = _random_batch(num_envs=4, steps=4, policy_dim=3, action_dim=2)
        metrics = policy.train_op(batch)
        for previous, parameter in zip(before, tracked, strict=True):
            self.assertFalse(torch.equal(previous, parameter))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["actor/approx_kl"])))
        self.assertEqual(metrics["actor/lr"], cfg.lr)

        restored = PPO(cfg, policy_dim=3, critic_dim=3, action_dim=2).optimizer
        restored.load_state_dict(optimizer.state_dict())
        self.assertTrue(restored.muon.state)
        self.assertTrue(restored.adamw.state)
        restored.param_groups[0]["lr"] = 0.001
        self.assertEqual(restored.adamw.param_groups[0]["lr"], 0.001)
        restored.param_groups[-1]["lr"] = 0.002
        self.assertEqual(restored.muon.param_groups[-1]["lr"], 0.002)

    def test_muon_falls_back_to_adamw_with_no_eligible_matrices(self):
        cfg = PPOConfig(muon=True, actor_hidden=(8,), critic_hidden=(8,))
        policy = PPO(cfg, policy_dim=3, critic_dim=3, action_dim=2)
        self.assertIsInstance(policy.optimizer, torch.optim.AdamW)

    def test_rejects_scalar_batches_without_feature_dimension(self):
        policy = PPO(PPOConfig(), policy_dim=3, critic_dim=3, action_dim=2)
        batch = _random_batch(num_envs=4, steps=4, policy_dim=3, action_dim=2)
        batch["action_log_prob"] = batch["action_log_prob"].squeeze(-1)
        with self.assertRaisesRegex(ValueError, "action_log_prob.*shape"):
            policy.train_op(batch)

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
                clamp_reward=True,
            ),
            policy_dim=3,
            critic_dim=3,
            action_dim=2,
        )
        batch = _random_batch(num_envs=4, steps=4, policy_dim=3, action_dim=2)
        raw = _aggregate_reward(batch)
        expected = batch["reward", "tracking"].sum(-1, keepdim=True)
        expected = expected + batch["reward", "regularization"]
        torch.testing.assert_close(raw, expected)
        scaled = policy._scaled_reward(raw, batch["done"])
        torch.testing.assert_close(scaled, expected.clamp_min(0) * 0.5)
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
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        )
        self.assertIn("actor/policy_loss", metrics)
        self.assertEqual(metrics["critic/neg_rew_ratio"], float((raw <= 0).float().mean()))

    def test_std_stays_inside_its_bounds(self):
        actor = NormalActor(4, 3, hidden=(8,), min_std=0.05, max_std=0.4)
        scale = actor.scale()
        self.assertTrue(torch.all(scale > 0.05))
        self.assertTrue(torch.all(scale < 0.4))


class _RecordingPolicy:
    """Write the same fields PPO would, without knowing about the collector."""

    def act(self, tensordict):
        observation = tensordict["observation"]
        tensordict["policy_observation"] = observation["state"]
        tensordict["critic_observation"] = torch.cat(
            [observation["state"], observation["extra"]], dim=-1
        )
        tensordict["action"] = torch.zeros(tensordict.batch_size[0], 2)
        tensordict["action_log_prob"] = torch.zeros(tensordict.batch_size[0], 1)
        return tensordict

    def record(self, tensordict, transition):
        observation = transition["observation"]
        tensordict["next_critic_observation"] = torch.cat(
            [observation["state"], observation["extra"]], dim=-1
        )


class _ScriptedEnvironment:
    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self._observation = torch.zeros(self.num_envs, 1)
        self._episode_step = torch.zeros(self.num_envs, 1, dtype=torch.long)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._observation[:] = 0
            self._episode_step[:] = 0
        else:
            self._observation[env_ids] = 50
            self._episode_step[env_ids] = 0
        return self._transition(
            torch.zeros(self.num_envs, 1),
            torch.zeros(self.num_envs, 1, dtype=torch.bool),
        )

    def step(self, action):
        self._observation = self._observation + 1
        self._episode_step += 1
        done = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        done[0, 0] = bool(self._observation[0, 0] == 1)
        return self._transition(torch.ones(self.num_envs, 1), done)

    def _transition(self, reward, done):
        return TensorDict(
            {
                "observation": TensorDict(
                    {
                        "state": self._observation.clone(),
                        "extra": self._observation.clone() + 9,
                    },
                    batch_size=[self.num_envs],
                ),
                "mdp_state": TensorDict(
                    {"episode_step": self._episode_step.clone()},
                    batch_size=[self.num_envs],
                ),
                "reward": {"tracking": reward},
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
            "reward": {
                "tracking": torch.randn(num_envs, steps, 2),
                "regularization": -torch.rand(num_envs, steps, 1),
            },
            "terminated": torch.zeros(num_envs, steps, 1, dtype=torch.bool),
            "done": torch.zeros(num_envs, steps, 1, dtype=torch.bool),
        },
        batch_size=[num_envs, steps],
    )


if __name__ == "__main__":
    unittest.main()
