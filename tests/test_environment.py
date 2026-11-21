"""CUDA regression tests for checkpoint replay and explicit resets."""

import unittest
from dataclasses import replace

import torch
import warp as wp
from tensordict import TensorDict

from newrl.configs import AnymalLocomotion
from newrl.env import StatefulEnvironment, StatelessEnvironment
from newrl.mdp.episode_ends import Timeout
from newrl.terrain import FlatMeshTerrainConfig


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class EnvironmentTests(unittest.TestCase):
    def test_mesh_terrain_spawn_replay_and_partial_reset(self):
        anchors = ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        cfg = AnymalLocomotion(
            num_envs=4,
            terrain=FlatMeshTerrainConfig(resolution=4, spawn_locations=anchors),
        )
        cfg.mdp = replace(cfg.mdp, root_xy_noise=0.0)
        env = StatelessEnvironment(cfg)
        sim_state, mdp_state, transition = env.reset()
        root_x = transition["physical_state", "robot", "body_pose_w"][:, 0, 0]
        self.assertTrue(torch.isin(root_x, root_x.new_tensor([-2.0, 2.0])).all())
        self.assertIn("ground_mesh", env.model.shape_label)
        repeat_sim, repeat_mdp, _ = env.reset()
        torch.testing.assert_close(repeat_sim, sim_state, atol=0, rtol=0)
        torch.testing.assert_close(repeat_mdp, mdp_state, atol=0, rtol=0)

        action = torch.zeros((4, env.action_dim), device=env.device)
        sim_state, mdp_state, transition = env.step(
            sim_state, mdp_state, TensorDict({"action": action}, batch_size=[4])
        )
        self.assertTrue(torch.isfinite(transition["observation", "proprio"]).all())
        reset_sim, reset_mdp, _ = env.reset(
            sim_state, mdp_state, env_ids=torch.tensor([0, 2])
        )
        torch.testing.assert_close(reset_sim[1::2], sim_state[1::2], atol=0, rtol=0)
        torch.testing.assert_close(reset_mdp[1::2], mdp_state[1::2], atol=0, rtol=0)

    def test_stateful_stateless_rollout_parity(self):
        for graph_capture, substeps in ((False, 4), (True, 4), (True, 3)):
            with self.subTest(graph_capture=graph_capture, substeps=substeps):
                self._check_rollout_parity(graph_capture, substeps)

    def _check_rollout_parity(self, graph_capture, substeps):
        cfg = AnymalLocomotion(
            num_envs=3,
            graph_capture=graph_capture,
            substeps=substeps,
            deterministic=True,
        )
        cfg.mdp = replace(cfg.mdp, hold_steps=4)
        cfg.mdp.truncations["timeout"] = Timeout(18)
        stateful = StatefulEnvironment(cfg)
        stateless = StatelessEnvironment(cfg)

        def compare(sim, mdp, actual, expected):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(stateful.mdp_state, mdp, atol=0, rtol=0)
            # Inspect all checkpoint fields without changing either environment.

            for name, value in sim.items():
                source = getattr(stateful.state, name, None)
                if source is None:
                    source = getattr(stateful.sim.mjw_data, name)
                torch.testing.assert_close(
                    wp.to_torch(source).reshape_as(value),
                    value,
                    atol=2e-4,
                    rtol=2e-4,
                    msg=lambda msg, field=name: f"{field}: {msg}",
                )

        sim, mdp, expected = stateless.reset()
        compare(sim, mdp, stateful.reset(), expected)
        for name, group in stateful.mdp.observations.items():
            self.assertIsInstance(group.shape, torch.Size)
            self.assertEqual(expected["observation", name].shape[1:], group.shape)
        phase = torch.arange(3 * stateful.action_dim, device=stateful.device).reshape(
            3, stateful.action_dim
        )
        for step in range(40):
            action = 0.1 * (phase + 0.17 * step).sin()
            actual = stateful.step(action)
            sim, mdp, expected = stateless.step(
                sim, mdp, TensorDict({"action": action}, batch_size=[3])
            )
            compare(sim, mdp, actual, expected)
            for key in ("reward", "terminated", "truncated", "done"):
                self.assertEqual(actual[key].shape, torch.Size([3, 1]))
            for group in actual["reward_terms"].values():
                for reward in group.values():
                    self.assertEqual(reward.shape, torch.Size([3, 1]))
            if step in (17, 27):
                self.assertEqual(
                    actual["truncated"].squeeze(-1).tolist(),
                    [True, False, True] if step == 17 else [False, True, False],
                )
            if step in (9, 17, 27, 35):
                ids = torch.tensor(
                    [0, 2] if step in (17, 35) else [1], device=stateful.device
                )
                sim, mdp, expected = stateless.reset(sim, mdp, env_ids=ids)
                compare(sim, mdp, stateful.reset(ids), expected)

    def test_checkpoint_replay_after_another_branch(self):
        cfg = AnymalLocomotion(num_envs=2, substeps=3, graph_capture=True)
        cfg.mdp = replace(cfg.mdp, hold_steps=25)
        env = StatelessEnvironment(cfg)
        sim, mdp, initial = env.reset()
        nominal = env.scene.entities["robot"].default_position
        displacement = initial["physical_state", "robot", "joint_pos"] - nominal
        self.assertTrue((displacement.abs() > 0).any())
        self.assertTrue(
            (displacement.abs() <= cfg.mdp.joint_position_noise + 1e-6).all()
        )
        replay_sim, replay_mdp, _ = env.reset()
        torch.testing.assert_close(replay_sim, sim, atol=0, rtol=0)
        torch.testing.assert_close(replay_mdp, mdp, atol=0, rtol=0)
        action = torch.full((2, env.action_dim), 0.1, device=env.device)
        inputs = TensorDict({"action": action}, batch_size=[2])
        # Advance through contact so the checkpoint includes solver warm-start state.
        for _ in range(24):
            sim, mdp, _ = env.step(sim, mdp, inputs)
        saved_sim, saved_mdp = sim.clone(), mdp.clone()

        next_sim, next_mdp, expected = env.step(sim, mdp, inputs)
        self.assertFalse(expected["done"].any())
        torch.testing.assert_close(
            expected["velocity_command"], mdp["velocity_command"]
        )
        self.assertFalse(
            torch.equal(next_mdp["velocity_command"], mdp["velocity_command"])
        )
        torch.testing.assert_close(
            expected["observation", "command"],
            next_mdp["velocity_command"],
        )
        other = TensorDict({"action": -action}, batch_size=[2])
        env.step(next_sim, next_mdp, other)
        actual_sim, actual_mdp, actual = env.step(sim, mdp, inputs)

        torch.testing.assert_close(sim, saved_sim, atol=0, rtol=0)
        torch.testing.assert_close(mdp, saved_mdp, atol=0, rtol=0)
        torch.testing.assert_close(actual_sim, next_sim, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(actual_mdp, next_mdp)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

        reset_sim, reset_mdp, _ = env.reset(
            next_sim, next_mdp, env_ids=torch.tensor([0])
        )
        self.assertEqual(reset_mdp["episode_step"].tolist(), [0, 25])
        torch.testing.assert_close(reset_sim[1], next_sim[1], atol=0, rtol=0)
        torch.testing.assert_close(reset_mdp[1], next_mdp[1], atol=0, rtol=0)
        self.assertNotEqual(
            reset_mdp["reset_rng"][0].item(), next_mdp["reset_rng"][0].item()
        )

    def test_stateful_step_returns_terminal_successor_without_reset(self):
        cfg = AnymalLocomotion(num_envs=2)
        cfg.mdp.truncations["timeout"] = Timeout(1)
        env = StatefulEnvironment(cfg)
        env.reset()
        action = torch.zeros((2, env.action_dim), device=env.device)
        transition = env.step(action)
        self.assertTrue(transition["done"].all())
        self.assertEqual(env.mdp_state["episode_step"].tolist(), [1, 1])

        reset = env.reset(torch.tensor([0]))
        self.assertEqual(env.mdp_state["episode_step"].tolist(), [0, 1])
        torch.testing.assert_close(
            reset["physical_state"][1], transition["physical_state"][1], atol=0, rtol=0
        )
