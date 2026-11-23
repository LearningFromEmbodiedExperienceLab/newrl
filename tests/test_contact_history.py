"""Contact history transitions and the air-time reward."""

import unittest

import torch
import warp as wp
from tensordict import TensorDict

from newrl.sensors.contact import _update_contact_history
from newrl.mdp.rewards import FeetAirTime


class ContactHistoryTests(unittest.TestCase):
    def test_substep_events_air_time_and_position(self):
        device = "cpu"
        force = wp.zeros(1, dtype=wp.vec3, device=device)
        positions = wp.array([[[1.0, 2.0, 3.0]]], dtype=wp.vec3, device=device)
        weights = wp.array([[2.0]], dtype=wp.float32, device=device)
        history = [
            wp.zeros(1, dtype=dtype, device=device)
            for dtype in (wp.bool, wp.bool, wp.float32, wp.float32, wp.vec3, wp.bool)
        ]

        def tick(load):
            wp.to_torch(force)[0] = torch.tensor([0.0, 0.0, load])
            wp.launch(
                _update_contact_history,
                dim=1,
                inputs=[force, positions, weights, 1.0, 0.1, True, True, *history],
                device=device,
            )

        tick(5.0)
        self.assertTrue(wp.to_torch(history[1]).item())
        self.assertEqual(wp.to_torch(history[3]).item(), 0.0)
        for value in history:
            value.zero_()
        tick(0.0)
        tick(0.0)
        self.assertAlmostEqual(wp.to_torch(history[2]).item(), 0.2)
        tick(5.0)
        self.assertTrue(wp.to_torch(history[1]).item())
        self.assertAlmostEqual(wp.to_torch(history[3]).item(), 0.2)
        torch.testing.assert_close(
            wp.to_torch(history[4])[0], torch.tensor([1.0, 2.0, 3.0])
        )
        self.assertTrue(wp.to_torch(history[5]).item())
        tick(0.0)
        self.assertTrue(wp.to_torch(history[1]).item())
        self.assertAlmostEqual(wp.to_torch(history[2]).item(), 0.1)
        self.assertAlmostEqual(wp.to_torch(history[3]).item(), 0.2)
        self.assertTrue(wp.to_torch(history[5]).item())

    def test_air_time_reward_strict_thresholds(self):
        reward = FeetAirTime("feet", weight=2.0, threshold_min=0.1, threshold_max=0.5)
        physical = TensorDict(
            {
                "feet": {
                    "current_air_time": torch.tensor(
                        [[0.0, 0.1, 0.2, 0.5], [0.15, 0.3, 0.4, 0.8]]
                    )
                }
            },
            batch_size=[2],
        )
        torch.testing.assert_close(reward(physical, None), torch.tensor([[2.0], [6.0]]))
