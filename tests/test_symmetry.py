import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from newrl.algos.genpo import GenPOConfig
from newrl.algos.ppo_symaug import PPOSymAugConfig
from newrl.mdp.base import ObservationGroup
from newrl.mdp.observations import velocity_command
from newrl.utils.symmetry import SymmetryTransform


class SymmetryTests(unittest.TestCase):
    def test_name_pairs_expand_in_observation_order(self):
        feet = SymmetryTransform.from_names(
            ("robot/RF_FOOT", "robot/LF_FOOT"),
            {"LF_FOOT": ("RF_FOOT", 1), "RF_FOOT": ("LF_FOOT", 1)},
            (1, -1, 1),
        )
        command = velocity_command.symmetry_transform(None)
        mirror = SymmetryTransform.cat([command, feet])
        observation = torch.arange(1.0, 10.0)
        expected = torch.tensor([1, -2, -3, 7, -8, 9, 4, -5, 6]).float()
        torch.testing.assert_close(mirror(observation), expected)
        torch.testing.assert_close(mirror(expected), observation)

    def test_rejects_incomplete_mirror_specs(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            SymmetryTransform((1, 0), (1, -1))
        with self.assertRaisesRegex(ValueError, "was not selected"):
            SymmetryTransform.from_names(("left",), {"left": ("right", 1)})
        group = ObservationGroup({"custom": SimpleNamespace(shape=torch.Size((1,)))})
        with self.assertRaisesRegex(ValueError, "custom.*does not declare a mirror"):
            group.symmetry_transform(None)

    def test_mirrored_actor_update_and_checkpoint(self):
        obs_keys = ("command",)
        group = ObservationGroup({"velocity": velocity_command})
        env = SimpleNamespace(
            mdp=SimpleNamespace(observations={"command": group}),
            scene=SimpleNamespace(
                action_symmetry_transform=lambda: SymmetryTransform((1, 0), (-1, -1))
            ),
            action_dim=2,
            device=torch.device("cpu"),
            observation_symmetry_transform=lambda groups: group.symmetry_transform(None),
        )
        for config_type in (PPOSymAugConfig, GenPOConfig):
            with self.subTest(algo=config_type.algo_id):
                cfg = config_type(
                    policy_obs_keys=obs_keys,
                    critic_obs_keys=obs_keys,
                    actor_hidden=(8,),
                    critic_hidden=(8,),
                    num_epochs=1,
                    num_minibatches=1,
                    symmetry_coef=1.0,
                )
                policy = cfg.build(env)
                generator = torch.Generator().manual_seed(4)
                shape = (2, 4)
                features = torch.randn(*shape, 3, generator=generator)
                batch = TensorDict(
                    {
                        "observation": {"command": features},
                        ("next", "reward"): {"tracking": features[..., :1]},
                        ("next", "done"): torch.zeros(*shape, 1, dtype=torch.bool),
                        ("next", "terminated"): torch.zeros(*shape, 1, dtype=torch.bool),
                    },
                    batch_size=shape,
                )
                policy.act(batch)
                batch["next", "observation"] = batch["observation"].clone()
                batch = policy.record(batch)
                norm_count = policy.policy_norm.count.clone()
                metrics = policy.train_op(batch)
                self.assertGreater(metrics["actor/symmetry_loss"], 0.0)
                self.assertLess(abs(metrics["actor/approx_kl"]), 1e-6)
                torch.testing.assert_close(policy.policy_norm.count, norm_count)
                restored = cfg.build(env)
                restored.load_training_state_dict(policy.training_state_dict())
                torch.testing.assert_close(restored.state_dict(), policy.state_dict())
