"""Known-motion checks for preprocessing, tracking terms, and MDP replay."""

import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from newrl.datasets.lafan1 import JOINT_NAMES, LAFAN1Dataset, _angular_velocity, _download
from newrl.mdp.base import ObservationGroup, RewardGroup
from newrl.mdp.episode_ends import ClipEnd
from newrl.mdp.motion_tracking import MotionTrackingMDP
from newrl.mdp.observations import ReferenceMotion
from newrl.mdp.rewards import ActionRate, TrackBodyMotion
from newrl.scene import RobotEntity
from newrl.utils.math import quat_rotate


class MotionTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "g1.xml"
        xml = '<mujoco><worldbody><body name="pelvis"><freejoint/><inertial pos="0.1 0 0" mass="1" diaginertia="1 1 1"/>'
        for joint in JOINT_NAMES:
            xml += f'<body name="{joint.replace("_joint", "_link")}" pos="0 0 0.1"><joint name="{joint}" axis="0 1 0"/><geom type="sphere" size="0.01" mass="0.1"/>'
        xml += '</body>' * (len(JOINT_NAMES) + 1) + '</worldbody></mujoco>'
        self.model.write_text(xml)
        self.source = self.directory / "csv"
        self.source.mkdir()
        t = np.arange(7) / 30
        raw = np.zeros((len(t), 36))
        raw[:, 0] = t
        raw[:, 2] = 0.8
        raw[:, 5] = np.sin(t / 2)
        raw[:, 6] = np.cos(t / 2)
        raw[:, 7:] = t[:, None] * 0.3
        raw[3:, 3:7] *= -1  # Equivalent quaternion signs must not cause a spin.
        for i in range(2):
            values = raw.copy()
            values[:, 0] += i * 10
            np.savetxt(self.source / f"clip{i}.csv", values, delimiter=",")
        self.motion = LAFAN1Dataset(self.source, model_path=self.model, cache_dir=self.directory / "cache")

    def scene(self):
        joints, bodies = len(JOINT_NAMES), len(self.motion.body_names)
        robot = RobotEntity(
            joint_target_indices=None, default_position=None,
            joint_names=JOINT_NAMES, body_names=self.motion.body_names,
            joint_q_indices=torch.arange(joints).expand(2, -1), joint_qd_indices=torch.arange(joints).expand(2, -1),
            root_q_indices=torch.arange(7).expand(2, -1), root_qd_indices=torch.arange(6).expand(2, -1),
            body_indices=torch.arange(bodies).expand(2, -1),
        )
        com = torch.zeros(bodies, 3)
        com[0, 0] = 0.1
        return SimpleNamespace(entities={"robot": robot}, control_dt=0.02,
                               model=SimpleNamespace(body_com=wp.array(com.numpy(), dtype=wp.vec3, device="cpu")))

    def test_resample_velocities_and_clip_clamping(self):
        self.assertEqual(self.motion.num_clips, 2)
        self.assertEqual(self.motion.data.batch_size, (22,))
        torch.testing.assert_close(self.motion.data["root_twist"][:, 0], torch.ones(22))
        torch.testing.assert_close(self.motion.data["root_twist"][:, 5], torch.ones(22))
        torch.testing.assert_close(self.motion.data["joint_vel"], torch.full((22, 29), 0.3))
        window = self.motion.get_slice(torch.tensor([0, 1]), torch.tensor([10, 0]), torch.tensor([0, 1, 99]))
        torch.testing.assert_close(window["root_pose"][0, :, 0], torch.full((3,), 0.2))
        torch.testing.assert_close(window["root_pose"][1, :, 0], torch.tensor([10., 10.02, 10.2]))
        torch.testing.assert_close(window["root_pose"][..., 3:].norm(dim=-1), torch.ones(2, 3))

    def test_env_var_and_bad_inputs(self):
        with patch.dict(os.environ, {"LAFAN1_DATASET": str(self.motion.path)}), patch("newrl.datasets.lafan1._download") as download:
            loaded = LAFAN1Dataset()
            download.assert_not_called()
            torch.testing.assert_close(loaded.data, self.motion.data)
        with self.assertRaisesRegex(ValueError, "frame rate"):
            LAFAN1Dataset(self.motion.path, fps=30)
        with self.assertRaises(FileNotFoundError):
            LAFAN1Dataset(self.directory / "missing")
        values = np.zeros((3, 36))
        np.savetxt(self.source / "bad.csv", values, delimiter=",")
        with self.assertRaisesRegex(ValueError, "quaternion"):
            LAFAN1Dataset(self.source, model_path=self.model, cache_dir=self.directory / "cache")

    def test_download_publishes_only_complete_files(self):
        target = self.directory / "download.csv"
        with patch("newrl.datasets.lafan1.urlopen", return_value=io.BytesIO(b"motion")):
            _download("https://example.com/motion", target)
        self.assertEqual(target.read_bytes(), b"motion")
        with patch("newrl.datasets.lafan1.urlopen", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                _download("https://example.com/motion", target)
        self.assertEqual(target.read_bytes(), b"motion")

    def test_mdp_reset_replay_partial_reset_and_reference_timing(self):
        scene = self.scene()
        mdp = MotionTrackingMDP(
            dataset=str(self.motion.path), goal_frames=2,
            observations={"command": ObservationGroup({"reference": ReferenceMotion(goal_frames=2)})},
            rewards={"tracking": RewardGroup({"root": TrackBodyMotion(1, .3, body_names="pelvis"), "action": ActionRate(.1)})},
            truncations={"clip_end": ClipEnd()},
        )._initialize(scene)
        initial = mdp.initial_state(2, 29, "cpu")
        state, pose = mdp.reset(initial, slice(None))
        again, pose_again = mdp.reset(initial, slice(None))
        torch.testing.assert_close(state, again)
        torch.testing.assert_close(pose["robot"], pose_again["robot"])
        physical = TensorDict({"robot": {"body_pose_w": state["reference", "body_pose_w"].clone()}}, [2])
        # COM velocity = origin velocity + omega cross rotated COM offset.
        root = pose["robot"]["root_pose"]
        offset = quat_rotate(root[..., 3:], torch.tensor([.1, 0, 0]).expand(2, -1))
        expected = state["reference", "root_twist"].clone()
        expected[..., :3] += torch.linalg.cross(expected[..., 3:], offset)
        torch.testing.assert_close(pose["robot"]["root_twist"], expected)
        partial, _ = mdp.reset(state, torch.tensor([0]))
        torch.testing.assert_close(partial[1], state[1])
        # Force the penultimate reference, then score against that reference.
        state["motion_t"] = (mdp._motion.clip_end[state["motion_id"].squeeze(-1)] - mdp._motion.clip_start[state["motion_id"].squeeze(-1)] - 2).unsqueeze(-1)
        mdp._refresh_reference(state)
        physical["robot", "body_pose_w"] = state["reference", "body_pose_w"].clone()
        original = state.clone()
        successor, transition = mdp.step(physical, state, torch.ones(2, 29))
        torch.testing.assert_close(state, original)
        torch.testing.assert_close(transition["reward", "tracking"], torch.tensor([[1., -2.9], [1., -2.9]]))
        self.assertTrue(transition["truncated"].all())
        self.assertFalse(transition["terminated"].any())
        self.assertEqual(transition["done"].shape, (2, 1))
        self.assertEqual(transition["observation", "command"].shape, (2, 85))
        self.assertTrue((successor["episode_step"] == 1).all())
        self.assertTrue((successor["previous_action"] == 1).all())
        next_again, output_again = mdp.step(physical, state, torch.ones(2, 29))
        torch.testing.assert_close(next_again, successor)
        torch.testing.assert_close(output_again, transition)

    def test_reward_conventions_anchoring_and_gradients(self):
        scene = self.scene()
        reference = self.motion.data[:2]
        pose = reference["body_pose_w"].clone().requires_grad_()
        com = torch.zeros_like(pose[..., :3])
        com[:, 0, 0] = .1
        twist = reference["body_twist_w"].clone()
        twist[..., :3] += torch.linalg.cross(twist[..., 3:], quat_rotate(pose[..., 3:], com))
        physical = TensorDict({"robot": {"body_pose_w": pose, "body_twist_w": twist}}, [2])
        state = TensorDict({"reference": reference}, [2])
        for quantity in ("position", "orientation", "linear_velocity", "angular_velocity"):
            term = TrackBodyMotion(2., .5, quantity)._initialize(scene)
            torch.testing.assert_close(term(physical, state), torch.full((2, 1), 2.))
        shifted = pose.detach().clone()
        shifted[..., 0] += 5
        physical["robot", "body_pose_w"] = shifted
        anchored = TrackBodyMotion(1, .3, anchor_body="pelvis")._initialize(scene)
        torch.testing.assert_close(anchored(physical, state), torch.ones(2, 1))
        physical["robot", "body_pose_w"] = pose + torch.tensor([.1, 0, 0, 0, 0, 0, 0])
        loss = TrackBodyMotion(1, .3)._initialize(scene)(physical, state).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(pose.grad).all())
        self.assertGreater(pose.grad.abs().sum(), 0)
        # Non-world batch shapes use the same tensor-only reward path.
        term = TrackBodyMotion(1, .3)._initialize(scene)
        self.assertEqual(term(physical.unsqueeze(1).expand(2, 3), state.unsqueeze(1).expand(2, 3)).shape, (2, 3, 1))


if __name__ == "__main__":
    unittest.main()
