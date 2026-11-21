"""CPU regression tests for MDP terms."""

import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from newrl.mdp.base import MDP, ObservationGroup, RewardGroup
from newrl.mdp.episode_ends import HeightTermination, TiltTermination
from newrl.mdp.observations import BodyHeight, ProjectedGravity
from newrl.mdp.rewards import JointAcc2, JointEnergy1, JointVel2, _JointTerm


class MDPTests(unittest.TestCase):
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
                "state": ObservationGroup(
                    {
                        "height": BodyHeight("robot", "base"),
                        "gravity": ProjectedGravity("robot", "base"),
                    }
                )
            },
            terminations={
                "height": HeightTermination(BodyHeight("robot", "robot/base")),
                "tilt": TiltTermination(ProjectedGravity("robot", "robot/base")),
            },
        )._initialize(scene)
        mdp_state = mdp.initial_state(1, 1, torch.device("cpu"))
        pose = torch.tensor([[[0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]]])
        gravity = torch.tensor([[[0.0, 0.0, -1.0]]])
        physical_state = TensorDict(
            {"robot": {"body_pose_w": pose, "projected_gravity": gravity}},
            batch_size=[1],
        )
        action = torch.zeros(1, 1)
        next_mdp_state, first = mdp.advance(physical_state, mdp_state, action)
        self.assertFalse(first["terminated"].item())
        pose[..., 2] = 0.2
        _, second = mdp.advance(physical_state, next_mdp_state, action)
        self.assertTrue(second["terminated"].item())
        torch.testing.assert_close(
            second["observation", "state"][:, :1], torch.tensor([[0.2]])
        )
        pose[..., 2] = 0.7
        gravity[..., 2] = 1.0
        self.assertFalse(mdp.terminations["height"](physical_state, mdp_state).item())
        self.assertTrue(mdp.terminations["tilt"](physical_state, mdp_state).item())
        mdp_state["spawn_height"] = torch.tensor([[0.5]])
        self.assertTrue(mdp.terminations["height"](physical_state, mdp_state).item())

    def test_reward_groups_keep_terms_and_sum_them(self):
        def reward(value):
            def term(physical_state, mdp_state):
                return torch.full((*mdp_state.batch_size, 1), value)

            return term

        mdp = MDP(
            rewards={
                "task": RewardGroup({"forward": reward(1.0), "yaw": reward(0.5)}),
                "regularization": RewardGroup({"action_rate": reward(-0.25)}),
            }
        )._initialize(SimpleNamespace())
        mdp_state = mdp.initial_state(2, 1, torch.device("cpu"))
        physical_state = TensorDict({}, batch_size=[2])
        _, transition = mdp.advance(physical_state, mdp_state, torch.zeros(2, 1))
        torch.testing.assert_close(transition["reward"], torch.full((2, 1), 1.25))
        torch.testing.assert_close(
            transition["reward_terms", "task", "yaw"], torch.full((2, 1), 0.5)
        )
        with self.assertRaises(ValueError):
            ObservationGroup({})


def _bind_joints(term, names=("LF_HAA", "LF_KFE")):
    entity = SimpleNamespace(
        joint_names=tuple(f"robot/{name}" for name in names),
        qd=torch.zeros(1, len(names), dtype=torch.long),
    )
    scene = SimpleNamespace(entities={"robot": entity}, control_dt=0.02)
    # Resolve indices without the actuator-gain tensors owned by JointEnergy1.
    return (
        term._initialize(scene)
        if isinstance(term, JointAcc2)
        else _JointTerm._initialize(term, scene)
    )


class JointPenaltyTests(unittest.TestCase):
    def test_velocity_acceleration_and_actuator_power(self):
        physical = TensorDict(
            {
                "robot": {
                    "joint_pos": torch.zeros(1, 2),
                    "joint_vel": torch.tensor([[1.0, -2.0]]),
                    "joint_torque": torch.tensor([[-1.0, 12.0]]),
                }
            },
            batch_size=[1],
        )
        mdp_state = TensorDict(
            {
                "action": torch.tensor([[0.0, 2.0]]),
                "episode_step": torch.tensor([2]),
                "previous_joint_vel": torch.tensor([[1.0, 0.0]]),
            },
            batch_size=[1],
        )
        velocity_penalty = _bind_joints(JointVel2(weight=0.5))(physical, mdp_state)
        torch.testing.assert_close(velocity_penalty, torch.tensor([[-2.5]]))
        hip_only = _bind_joints(JointVel2(weight=0.5, joint_names=".*HAA"))(
            physical, mdp_state
        )
        torch.testing.assert_close(hip_only, torch.tensor([[-0.5]]))

        acceleration_penalty = _bind_joints(JointAcc2(weight=1.0))(physical, mdp_state)
        # Squared change is 0 and 4. The carried velocity becomes the current one.
        torch.testing.assert_close(acceleration_penalty, torch.tensor([[-10000.0]]))
        torch.testing.assert_close(
            mdp_state["previous_joint_vel"], torch.tensor([[1.0, -2.0]])
        )
        mdp_state["episode_step"] = torch.tensor([1])
        opening = _bind_joints(JointAcc2(weight=1.0))(physical, mdp_state)
        torch.testing.assert_close(opening, torch.zeros(1, 1))

        energy = _bind_joints(JointEnergy1(weight=1.0))
        power = energy(physical, mdp_state)
        torch.testing.assert_close(power, torch.tensor([[-25.0]]))
