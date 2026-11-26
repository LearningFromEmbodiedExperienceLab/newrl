"""Robot bundle import and actuator coverage."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import newton

from newrl.assets.robots import ActuatorGroup, RobotConfig, assetx_as2, assetx_g1, assetx_spot, scalar_joints


class ActuatorSpecTests(unittest.TestCase):
    def test_passive_damping_override_preserves_pd_gain(self):
        def import_joints(builder, *args, **kwargs):
            for name in ("hip", "knee"):
                body = builder.add_link()
                builder.add_joint_revolute(
                    parent=-1, child=body, label=name,
                    axis=newton.ModelBuilder.JointDofConfig(damping=0.7),
                )

        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "robot.usda"
            asset.touch()
            for override in (None, 0.0, 1.2):
                with self.subTest(override=override), patch.object(
                    newton.ModelBuilder, "add_usd", import_joints
                ):
                    config = RobotConfig(
                        asset=asset, position=(0, 0, 0),
                        joint_position={".*": 0.0}, action_scale=0.5,
                        actuators=(
                            ActuatorGroup(("hip",), 10.0, 2.0, 0.1, viscous_damping=override),
                            ActuatorGroup(("knee",), 10.0, 3.0, 0.1),
                        ),
                    )
                    builder = config.build()
                    self.assertEqual(builder.joint_target_kd, [2.0, 3.0])
                    self.assertEqual(builder.joint_damping, [0.7 if override is None else override, 0.7])
            with patch.object(newton.ModelBuilder, "add_usd", import_joints):
                config = RobotConfig(
                    asset=asset, position=(0, 0, 0), joint_position=(0, 0),
                    action_scale=0.5, stiffness=(10, 10), damping=(2, 3), armature=(0.1, 0.1),
                )
                self.assertEqual(config.build().joint_damping, [0.7, 0.7])

    def test_passive_damping_rejects_invalid_values(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ActuatorGroup((".*",), 10.0, 2.0, 0.1, viscous_damping=value)

    def test_g1_covers_all_29_joints(self):
        config = assetx_g1()
        builder = config.build()
        joints = scalar_joints(builder)
        labels = [builder.joint_label[joint] for joint in joints]
        self.assertEqual(len(joints), 29)
        stiffness, damping, armature, effort, speed, _, _ = config._gains(labels)
        scales = config.action_scales(labels)
        for index, joint in enumerate(joints):
            dof = builder.joint_qd_start[joint]
            self.assertAlmostEqual(builder.joint_target_ke[dof], stiffness[index])
            self.assertAlmostEqual(builder.joint_target_kd[dof], damping[index])
            self.assertAlmostEqual(builder.joint_armature[dof], armature[index])
            self.assertAlmostEqual(scales[index], 0.25 * effort[index] / stiffness[index])
            self.assertGreater(speed[index], 0)
            label = labels[index].rsplit("/", 1)[-1]
            if label.endswith("knee_joint"):
                self.assertAlmostEqual(builder.joint_q[builder.joint_q_start[joint]], 0.669)
        # Root placement must come from RobotConfig, without adding the XML height.
        self.assertAlmostEqual(builder.joint_q[2], 0.76, places=5)

    def test_spot_and_as2_cover_their_joints(self):
        for factory, hip_stiffness, calf_armature, scale in (
            (assetx_spot, 60.0, 0.01, 0.5),
            (assetx_as2, 60.0, 0.038, 0.4),
        ):
            with self.subTest(factory=factory.__name__):
                config = factory()
                builder = config.build()
                joints = scalar_joints(builder)
                labels = [builder.joint_label[joint] for joint in joints]
                self.assertEqual(len(labels), 12)
                stiffness, _, armature, effort, _, _, _ = config._gains(labels)
                scales = config.action_scales(labels)
                self.assertEqual(set(scales), {scale})
                self.assertIn(hip_stiffness, stiffness)
                self.assertIn(calf_armature, armature)
                self.assertTrue(all(value is not None for value in effort))


if __name__ == "__main__":
    unittest.main()
