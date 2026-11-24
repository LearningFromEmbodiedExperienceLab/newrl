"""Verify camera refresh history across selective resets and replay."""

import unittest
from dataclasses import replace
from math import pi

import torch
from tensordict import TensorDict

from newrl.configs import AS2WLocomotion
from newrl.env import StatelessEnvironment
from newrl.mdp.base import ObservationGroup
from newrl.mdp.observations import DepthCameraObservation
from newrl.sensors import DepthCameraConfig
from newrl.sensors.camera import camera_refresh_mask


class CameraClockTests(unittest.TestCase):
    def test_twenty_hz_at_fifty_control_steps_per_second(self):
        phase = torch.full((2,), 5)
        captures = [[], []]
        for step in range(1, 51):
            due = camera_refresh_mask(phase, 5, tick=2, reset=False, reset_mask=None)
            for world in range(2):
                if due[world]:
                    captures[world].append(step)
            if step == 2:
                camera_refresh_mask(phase, 5, tick=2, reset=True, reset_mask=torch.tensor([False, True]))
        self.assertEqual(captures[0], [v for k in range(0, 50, 5) for v in (k + 3, k + 5)])
        self.assertEqual(captures[1], [v + 2 for v in captures[0] if v + 2 <= 50])


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
            output_physical_fields=(
                ("front", "depth"), ("front", "valid"),
                ("short_range", "depth"), ("short_range", "valid"),
            ),
        )
        config.sensors["short_range"] = replace(config.sensors["front"], max_depth=0.1)
        config.mdp.observations["depth"] = ObservationGroup(
            {"front": DepthCameraObservation("front")}
        )
        env = StatelessEnvironment(config)
        action = torch.zeros((2, env.action_dim), device=env.device)
        inputs = TensorDict({"action": action}, batch_size=[2])
        sim, mdp, initial = env.reset()
        depth_key = ("physical_state", "front", "depth")
        self.assertEqual(sim["camera_depth:front"].dtype, torch.float16)
        self.assertEqual(initial[depth_key].dtype, torch.float16)
        self.assertEqual(initial["observation", "depth"].dtype, torch.float16)
        self.assertTrue(initial["physical_state", "front", "valid"].all().item())
        self.assertFalse(initial["physical_state", "short_range", "valid"].any().item())
        self.assertFalse(initial["physical_state", "short_range", "depth"].any().item())

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
