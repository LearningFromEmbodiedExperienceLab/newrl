"""Regex actuator coverage for the Spot and AS2 bundles."""

import unittest

from newrl.robots import assetx_as2, assetx_spot, scalar_joints


class ActuatorSpecTests(unittest.TestCase):
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
                stiffness, _, armature, effort, _, _ = config._gains(labels)
                scales = config.action_scales(labels)
                self.assertEqual(set(scales), {scale})
                self.assertIn(hip_stiffness, stiffness)
                self.assertIn(calf_armature, armature)
                self.assertTrue(all(value is not None for value in effort))


if __name__ == "__main__":
    unittest.main()
