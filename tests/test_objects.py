"""Scene assembly, passive object state, and heterogeneous solver replay."""

import unittest

import torch
import warp as wp
from tensordict import TensorDict

from newrl.configs import AS2Locomotion, EnvironmentConfig
from newrl.env import StatefulEnvironment, StatelessEnvironment
from newrl.utils.config import cs
from newrl.assets.objects import BoxConfig, ChairConfig, DoorConfig, DrawerConfig, ObjectVariants, SphereConfig
from newrl.randomization import DomainRandomization, DomainRandomizationConfig, ShapeFriction
from newrl.scene import Scene
from newrl.sensors import ContactSensorConfig
from newrl.solver_adapters import MujocoSimConfig, SteelSimConfig


class ObjectTests(unittest.TestCase):
    def test_incompatible_topology(self):
        with self.assertRaisesRegex(ValueError, "fixture.*variant 1.*body_label"):
            Scene(EnvironmentConfig(device="cpu", objects={
                "fixture": ObjectVariants((BoxConfig(), DoorConfig())),
            }))

    def test_heterogeneous_layout(self):
        config = EnvironmentConfig(
            num_envs=4, device="cpu",
            objects={"prop": ObjectVariants((SphereConfig(), ChairConfig())), "drawer": DrawerConfig()},
        )
        scene = Scene(config)
        self.assertEqual(scene.object_variant_ids, Scene(config).object_variant_ids)
        self.assertEqual(set(scene.object_variant_ids["prop"]), {0, 1})
        drawer = scene.entities["drawer"]
        self.assertEqual(drawer.find_joints("slide")[0], [0])
        self.assertEqual(drawer.find_bodies("tray")[0], [1])
        self.assertEqual(drawer.joint_q_indices.shape, (4, 1))
        self.assertFalse(scene.robots)

    def test_friction_follows_ragged_shape_ownership(self):
        scene = Scene(EnvironmentConfig(num_envs=4, device="cpu", objects={
            "prop": ObjectVariants((SphereConfig(), ChairConfig())),
        }))
        randomization = DomainRandomization(scene, DomainRandomizationConfig(
            friction=ShapeFriction(entity="prop", body_names="base", bounds=(0.2, 0.8)),
        ))
        friction = wp.to_torch(scene.model.shape_material_mu)
        owner = wp.to_torch(scene.model.shape_body)
        for world, (body,) in enumerate(scene.entities["prop"].body_indices):
            actual = friction[owner == body]
            samples = randomization.values["shape_friction"][world]
            torch.testing.assert_close(actual, samples[:len(actual)])
            self.assertTrue((samples[len(actual):] == 0).all())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tabletop_survival(self):
        env = StatefulEnvironment(cs.get("task", "G1Tabletop"))
        initial = env.reset()["observation", "privileged"]
        self.assertEqual(set(env.scene.object_variant_ids), {"object"})
        self.assertEqual(set(env.scene.object_variant_ids["object"]), {0, 1, 2})
        inputs = TensorDict({"action": torch.zeros(env.num_envs, env.action_dim, device=env.device)},
                            batch_size=[env.num_envs])
        for _ in range(50):
            step = env.step(inputs)
            self.assertEqual(list(step["reward"].keys()), ["survival"])
            reward = step["reward", "survival"]
            torch.testing.assert_close(reward, torch.ones(env.num_envs, 1, device=env.device))
            torch.testing.assert_close(step["observation", "privileged"][:, :3], initial[:, :3],
                                       atol=0.01, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_partial_reset_and_replay(self):
        for solver in (MujocoSimConfig(), SteelSimConfig()):
            with self.subTest(solver=type(solver).__name__):
                self._check_solver(solver)

    def _check_solver(self, solver):
        variant = (
            ChairConfig(position=(2, 0, 1)) if isinstance(solver, SteelSimConfig)
            else BoxConfig(half_extents=(0.2, 0.3, 0.15), position=(2, 0, 1))
        )
        config = AS2Locomotion(
            num_envs=3, deterministic=True, solver=solver, graph_capture=False,
            objects={
                "prop": ObjectVariants((BoxConfig(position=(2, 0, 1)), variant)),
                "door": DoorConfig(position=(3, 0, 0)),
            },
            output_physical_fields=(("prop", "body_pose_w"), ("door", "joint_pos"),
                                    ("touch", "force_w")),
        )
        config.sensors["touch"] = ContactSensorConfig("robot", ".*_foot", counterpart="prop")
        env = StatelessEnvironment(config)
        self.assertEqual(env.action_dim, len(env.scene.robots["robot"].joint_names))
        sim, mdp, initial = env.reset()
        inputs = TensorDict({"action": torch.zeros(3, env.action_dim, device=env.device)}, [3])
        after, memory, step = env.step(sim, mdp, inputs)
        self.assertTrue((step["physical_state", "prop", "body_pose_w"][:, 0, 2] < 1).all())
        again, _, _ = env.step(sim, mdp, inputs)
        torch.testing.assert_close(after["joint_q"], again["joint_q"])
        reset, _, output = env.reset(after, memory, env_ids=torch.tensor([1], device=env.device))
        torch.testing.assert_close(reset["joint_q"][[0, 2]], after["joint_q"][[0, 2]])
        torch.testing.assert_close(output["physical_state","prop","body_pose_w"][1],
                                   initial["physical_state","prop","body_pose_w"][1])
