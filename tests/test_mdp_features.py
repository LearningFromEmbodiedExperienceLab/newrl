"""CPU regression for sharing physical features within an MDP transition."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from newrl.mdp.base import MDP
from newrl.mdp.episode_ends import Fall
from newrl.mdp.observations import BodyHeight, ProjectedGravity


class StepFeatureTests(unittest.TestCase):
    def test_fall_and_observations_share_only_current_physical_features(self):
        scene = SimpleNamespace(
            entities={
                "robot": SimpleNamespace(
                    body_names=("robot/base",),
                    bodies=torch.zeros(1, 1, dtype=torch.long),
                )
            }
        )
        mdp = MDP(
            observations={
                "height": BodyHeight("robot", "base"),
                "gravity": ProjectedGravity("robot", "base"),
            },
            terminations={
                "fall": Fall(
                    BodyHeight("robot", "robot/base"),
                    ProjectedGravity("robot", "robot/base"),
                )
            },
        )._initialize(scene)
        mdp_state = mdp.initial_state(1, 1, torch.device("cpu"))
        pose = torch.tensor([[[0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]]])
        physical_state = TensorDict({"robot": {"body_pose_w": pose}}, batch_size=[1])
        action = torch.zeros(1, 1)
        with (
            patch.object(
                BodyHeight, "_compute", autospec=True, side_effect=BodyHeight._compute
            ) as height,
            patch.object(
                ProjectedGravity,
                "_compute",
                autospec=True,
                side_effect=ProjectedGravity._compute,
            ) as gravity,
        ):
            next_mdp_state, first = mdp.advance(physical_state, mdp_state, action)
            self.assertFalse(first["terminated"].item())
            self.assertEqual(height.call_count, 1)
            self.assertEqual(gravity.call_count, 1)

            pose[..., 2] = 0.2
            _, second = mdp.advance(physical_state, next_mdp_state, action)
            self.assertTrue(second["terminated"].item())
            torch.testing.assert_close(
                second["observation", "height"], torch.tensor([[0.2]])
            )
            self.assertEqual(height.call_count, 2)
            self.assertEqual(gravity.call_count, 2)

            mdp.observations.clear()
            _, third = mdp.advance(physical_state, next_mdp_state, action)
            self.assertTrue(third["terminated"].item())
            self.assertEqual(height.call_count, 3)
            self.assertEqual(gravity.call_count, 3)
