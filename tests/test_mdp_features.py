"""CPU regression tests for MDP terms."""

import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from newrl.scene import RobotEntity
from newrl.mdp.base import MDP, ObservationGroup, RewardGroup
from newrl.mdp.episode_ends import HeightTermination, TiltTermination, Timeout
from newrl.mdp.observations import BodyHeight, BodyLinearVelocity, ProjectedGravity
from newrl.mdp.rewards import (
    ActionRate,
    BodyAngularVelocity2,
    BodyLinearVelocity2,
    FeetAirTime,
    FootSlip,
    JointAcc2,
    JointEnergy1,
    JointVel2,
    TrackVelocity,
    _JointTerm,
)


class MDPTests(unittest.TestCase):
    def test_history_updates_only_on_step_and_retains_imagination_gradients(self):
        robot = RobotEntity(
            joint_target_indices=None, body_indices=None, root_q_indices=None, default_position=None,
            joint_q_indices=torch.zeros(1, 2, dtype=torch.long), joint_qd_indices=torch.zeros(1, 2, dtype=torch.long),
            joint_names=('robot/a', 'robot/b'), body_names=(),
        )
        mdp = MDP(
            rewards={'regularization': RewardGroup({'acceleration': JointAcc2(1.)})},
            truncations={'timeout': Timeout(3)},
        )
        self.assertIn('previous_joint_vel', mdp.replay_fields)
        mdp = mdp._initialize(SimpleNamespace(entities={'robot': robot}, control_dt=.5))
        initial = mdp.initial_state(2, 2, torch.device('cpu'))
        first_velocity = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        physical = TensorDict({'robot': {'joint_vel': first_velocity}}, [2])
        state, transition = mdp.step(physical, initial, torch.zeros(2, 2))
        torch.testing.assert_close(transition['reward', 'regularization'], torch.zeros(2, 1))
        torch.testing.assert_close(state['previous_joint_vel'], first_velocity)
        self.assertIn('previous_joint_vel', mdp.replay_fields)
        torch.testing.assert_close(initial['previous_joint_vel'], torch.zeros(2, 2))
        physical = TensorDict({'robot': {'joint_vel': first_velocity.detach() + torch.tensor([1., 2.])}}, [2])
        decision = state.clone()
        decision['action'] = torch.zeros(2, 2)
        original = decision.clone()
        for compute in (mdp.compute_reward, mdp.compute_observation, mdp.compute_termination, mdp.compute_truncation):
            first, second = compute(physical, decision), compute(physical, decision)
            torch.testing.assert_close(first, second)
            torch.testing.assert_close(decision, original)
        reward = mdp.compute_reward(physical, decision)['regularization']
        torch.testing.assert_close(reward, torch.full((2, 1), -20.))
        reward.sum().backward()
        torch.testing.assert_close(first_velocity.grad, torch.tensor([[8., 16.], [8., 16.]]))
        reset, _ = mdp.reset(state, torch.tensor([0]))
        torch.testing.assert_close(reset['previous_joint_vel'][0], torch.zeros(2))
        torch.testing.assert_close(reset[1], state[1])

    def test_velocity_rewards_use_old_command_and_endings_use_new_command(self):
        from newrl.mdp.velocity import VelocityMDP

        def command(physical, state):
            return state['velocity_command']
        command.shape = torch.Size((3,))
        command.physical_fields = frozenset()
        def reward(physical, state):
            return -(physical['velocity'] - state['velocity_command']).square().sum(-1, keepdim=True)
        def ending(physical, state):
            return state['velocity_command'][..., :1] < 2
        mdp = VelocityMDP(
            hold_steps=1, observations={'command': ObservationGroup({'command': command})},
            rewards={'tracking': RewardGroup({'tracking': reward})}, terminations={'new_command': ending},
        )
        state = mdp.initial_state(2, 2, torch.device('cpu'))
        state['velocity_command'] = torch.tensor([[7., 8., 9.], [7., 8., 9.]])
        state['command_steps_left'][:] = 1
        original = state.clone()
        successor, transition = mdp.step(TensorDict({'velocity': state['velocity_command'].clone()}, [2]), state, torch.ones(2, 2))
        torch.testing.assert_close(state, original)
        torch.testing.assert_close(transition['reward', 'tracking'], torch.zeros(2, 1))
        torch.testing.assert_close(transition['velocity_command'], state['velocity_command'])
        torch.testing.assert_close(transition['observation', 'command'], successor['velocity_command'])
        self.assertTrue(transition['terminated'].all())
        self.assertTrue((successor['episode_step'] == 1).all())

    def test_scalar_shapes_for_multidimensional_batches(self):
        mdp = MDP(truncations={"timeout": Timeout(3)})
        state = TensorDict(
            {
                "episode_step": torch.tensor([[[2], [3], [4]], [[1], [2], [3]]]),
                "previous_action": torch.zeros(2, 3, 2),
            },
            batch_size=[2, 3],
        )
        physical = TensorDict({}, batch_size=[2, 3])
        original = state.clone()
        successor, result = mdp.step(physical, state, torch.zeros(2, 3, 2))
        self.assertEqual(list(result["reward"].keys()), [])
        for key in ("terminated", "truncated", "done"):
            self.assertEqual(result[key].shape, (2, 3, 1))
        torch.testing.assert_close(result["truncated"], successor["episode_step"] >= 3)
        torch.testing.assert_close(successor["episode_step"], state["episode_step"] + 1)
        torch.testing.assert_close(state, original)

    def test_fall_and_observations_share_only_current_physical_features(self):
        scene = SimpleNamespace(
            entities={
                "robot": RobotEntity(
                    joint_names=(), joint_q_indices=None, joint_qd_indices=None, joint_target_indices=None, root_q_indices=None,
                    default_position=None,
                    body_names=("robot/base",),
                    body_indices=torch.zeros(1, 1, dtype=torch.long),
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
        next_mdp_state, first = mdp.step(physical_state, mdp_state, action)
        self.assertFalse(first["terminated"].item())
        pose[..., 2] = 0.2
        _, second = mdp.step(physical_state, next_mdp_state, action)
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

    def test_reward_groups_concatenate_terms_in_definition_order(self):
        def reward(value):
            def term(physical_state, mdp_state):
                return torch.full((*mdp_state.batch_size, 1), value)

            term.range = (value, value)
            return term

        mdp = MDP(
            rewards={
                "task": RewardGroup({"forward": reward(1.0), "yaw": reward(0.5)}),
                "regularization": RewardGroup({"action_rate": reward(-0.25)}),
            }
        )._initialize(SimpleNamespace())
        mdp_state = mdp.initial_state(2, 1, torch.device("cpu"))
        physical_state = TensorDict({}, batch_size=[2])
        _, transition = mdp.step(physical_state, mdp_state, torch.zeros(2, 1))
        torch.testing.assert_close(
            transition["reward", "task"], torch.tensor([[1.0, 0.5], [1.0, 0.5]])
        )
        torch.testing.assert_close(
            transition["reward", "regularization"], torch.full((2, 1), -0.25)
        )
        with self.assertRaises(ValueError):
            ObservationGroup({})


class RewardRangeTests(unittest.TestCase):
    def test_body_velocity_penalties_select_bodies_and_frame_components(self):
        batch_size = (2, 3)
        scene = SimpleNamespace(
            entities={
                "robot": RobotEntity(
                    joint_names=(), joint_q_indices=None, joint_qd_indices=None, joint_target_indices=None, root_q_indices=None,
                    default_position=None,
                    body_names=("robot/base", "robot/ignored", "robot/foot"),
                    body_indices=torch.zeros(2, 3, dtype=torch.long),
                )
            }
        )
        linear = BodyLinearVelocity2(
            "robot", ("base", "foot"), weight=2.0, axis=(0.0, 0.0, 1.0), frame="body"
        )._initialize(scene)
        angular = BodyAngularVelocity2(
            "robot", "base", weight=0.5, axis=(1.0, 1.0, 0.0), frame="body"
        )._initialize(scene)
        world = BodyLinearVelocity2(
            "robot", "base", weight=2.0, axis=(0.0, 0.0, 1.0)
        )._initialize(scene)
        twist = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [100] * 6, [3, 4, 5, 0, 0, 0]],
            dtype=torch.float32,
        ).expand(*batch_size, -1, -1)
        pose = torch.zeros(*batch_size, 3, 7)
        pose[..., 3] = 2 ** -0.5
        pose[..., 6] = 2 ** -0.5
        physical = TensorDict(
            {"robot": {"body_twist_w": twist, "body_pose_w": pose}},
            batch_size=batch_size,
        )
        torch.testing.assert_close(
            linear(physical, None), torch.full((*batch_size, 1), -40.0)
        )
        torch.testing.assert_close(
            angular(physical, None), torch.full((*batch_size, 1), -26.0)
        )
        torch.testing.assert_close(
            world(physical, None), torch.full((*batch_size, 1), -18.0)
        )
        self.assertEqual(world.physical_fields, {("robot", "body_twist_w")})
        self.assertEqual(
            linear.physical_fields, {("robot", "body_twist_w"), ("robot", "body_pose_w")}
        )
        self.assertEqual(angular.range, (-float("inf"), 0.0))
        self.assertEqual(
            BodyLinearVelocity2("robot", "base", -1.0).range, (0.0, float("inf"))
        )
        self.assertEqual(
            BodyAngularVelocity2("robot", "base", 1.0, axis=(0.0, 0.0, 0.0)).range,
            (0.0, 0.0),
        )

    def test_penalty_ranges_follow_weight_sign(self):
        for penalty in (ActionRate, JointAcc2, JointEnergy1, JointVel2):
            self.assertEqual(penalty(weight=2.0).range, (-float("inf"), 0.0))
            self.assertEqual(penalty(weight=-2.0).range, (0.0, float("inf")))
            self.assertEqual(penalty(weight=0.0).range, (0.0, 0.0))
        self.assertEqual(FootSlip("feet", weight=2.0).range, (-float("inf"), 0.0))

    def test_tracking_and_bound_air_time_ranges(self):
        velocity = BodyLinearVelocity("robot", "base")
        self.assertEqual(TrackVelocity(velocity, slice(2), 0.5, 1.0).range, (0.0, 0.5))
        self.assertEqual(TrackVelocity(velocity, slice(2), -0.5, 1.0).range, (-0.5, 0.0))
        scene = SimpleNamespace(
            resolve_physical_fields=lambda _: None,
            sensors={"feet": SimpleNamespace(body_ids=torch.zeros(2, 4))},
        )
        air_time = FeetAirTime("feet", weight=0.5)
        self.assertEqual(air_time.range, (0.0, float("inf")))
        self.assertEqual(air_time._initialize(scene).range, (0.0, 2.0))
        self.assertEqual(FeetAirTime("feet", weight=-0.5)._initialize(scene).range, (-2.0, 0.0))


def _bind_joints(term, names=("LF_HAA", "LF_KFE")):
    entity = RobotEntity(
        joint_q_indices=None, joint_target_indices=None, body_indices=None, root_q_indices=None, default_position=None,
        body_names=(),
        joint_names=tuple(f"robot/{name}" for name in names),
        joint_qd_indices=torch.zeros(1, len(names), dtype=torch.long),
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
                "episode_step": torch.tensor([[2]]),
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
        # Squared change is 0 and 4. Evaluation must leave history unchanged.
        torch.testing.assert_close(acceleration_penalty, torch.tensor([[-10000.0]]))
        torch.testing.assert_close(
            mdp_state["previous_joint_vel"], torch.tensor([[1.0, 0.0]])
        )
        mdp_state["episode_step"] = torch.tensor([[0]])
        opening = _bind_joints(JointAcc2(weight=1.0))(physical, mdp_state)
        torch.testing.assert_close(opening, torch.zeros(1, 1))

        energy = _bind_joints(JointEnergy1(weight=1.0))
        power = energy(physical, mdp_state)
        torch.testing.assert_close(power, torch.tensor([[-25.0]]))
