"""Verify camera refresh history across selective resets and replay."""

import unittest
from math import pi

import torch
from tensordict import TensorDict

from newrl.configs import AS2WLocomotion
from newrl.env import StatelessEnvironment
from newrl.sensors import DepthCameraConfig


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DepthCameraTests(unittest.TestCase):
    def test_refresh_schedule_survives_partial_reset_and_replay(self):
        config = AS2WLocomotion(
            sphere_wheels=True,
            num_envs=2,
            sensors={
                "front": DepthCameraConfig(
                    width=8, height=6, pos=(1.5, 0.0, 0.3),
                    rpy=(0.0, pi / 2, 0.0), update_every=3,
                )
            },
            output_physical_fields=(("front", "depth"), ("front", "valid")),
        )
        env = StatelessEnvironment(config)
        action = torch.zeros((2, env.action_dim), device=env.device)
        inputs = TensorDict({"action": action}, batch_size=[2])
        sim, mdp, initial = env.reset()
        depth_key = ("physical_state", "front", "depth")
        self.assertTrue(initial["physical_state", "front", "valid"].all().item())

        for _ in range(2):
            sim, mdp, held = env.step(sim, mdp, inputs)
            torch.testing.assert_close(held[depth_key], initial[depth_key], atol=0, rtol=0)

        indices = torch.tensor([0], device=env.device)
        sim, mdp, partial = env.reset(sim, mdp, env_ids=indices)
        torch.testing.assert_close(partial[depth_key][1], held[depth_key][1], atol=0, rtol=0)

        later_sim, later_mdp, refreshed = env.step(sim, mdp, inputs)
        torch.testing.assert_close(refreshed[depth_key][0], partial[depth_key][0], atol=0, rtol=0)
        self.assertFalse(torch.equal(refreshed[depth_key][1], held[depth_key][1]))

        env.step(later_sim, later_mdp, inputs)
        _, _, replayed = env.step(sim, mdp, inputs)
        torch.testing.assert_close(replayed[depth_key], refreshed[depth_key], atol=0, rtol=0)
