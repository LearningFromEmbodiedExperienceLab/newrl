"""Construction, Newton FK equivalence and gradients for SuperMimic state plumbing."""

from types import SimpleNamespace
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import newton
import torch
from tensordict import TensorDict
import warp as wp

from newrl.algos.supermimic import GeneralizedCoordinatesConfig, SuperMimicConfig
from newrl.configs import G1MotionTracking
from newrl.mdp.base import MDP


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        builder = newton.ModelBuilder()
        root = builder.add_link(label='pelvis', com=wp.vec3(.1, .03, -.04))
        builder.add_joint_free(child=root, label='root')
        arm = builder.add_link(label='arm', com=wp.vec3(.2, -.04, .02))
        fixed = builder.add_link(label='fixed', com=wp.vec3(.02, .1, .03))
        slider = builder.add_link(label='slider', com=wp.vec3(.03, .02, .1))
        hinge = builder.add_joint_revolute(
            parent=root, child=arm, axis=wp.vec3(0, 1, 0), label='hinge',
            parent_xform=wp.transform(wp.vec3(.1, .2, .3), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), .3)),
            child_xform=wp.transform(wp.vec3(-.1, .1, -.1), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -.4)),
        )
        builder.add_joint_fixed(parent=arm, child=fixed, parent_xform=wp.transform(wp.vec3(.2, 0, .1), wp.quat_identity()))
        slide = builder.add_joint_prismatic(parent=fixed, child=slider, axis=wp.vec3(1, 0, 0), label='slide', child_xform=wp.transform(wp.vec3(0, .1, .2), wp.quat_identity()))
        builder.add_articulation(list(range(builder.joint_count)))
        model = builder.finalize(device='cpu')
        q_indices = torch.tensor([builder.joint_q_start[j] for j in (hinge, slide)])
        robot = SimpleNamespace(
            root_q=torch.arange(7).unsqueeze(0), bodies=torch.arange(4).unsqueeze(0),
            q=q_indices.unsqueeze(0), qd=q_indices.unsqueeze(0), joint_names=('hinge', 'slide'),
            body_names=('pelvis', 'arm', 'fixed', 'slider'),
        )
        self.scene = SimpleNamespace(model=model, entities={'robot':robot})
        self.representation = GeneralizedCoordinatesConfig().build(self.scene)
        rows = []
        for i in range(3):
            state = model.state()
            q = wp.to_torch(state.joint_q)
            qd = wp.to_torch(state.joint_qd)
            q[:3] = torch.tensor([i * .3, -.2, 1.])
            q[3:7] = torch.tensor([.1, -.2, .3, .9]); q[3:7] /= q[3:7].norm()
            q[q_indices] = torch.tensor([.3 + i * .1, .2 - i * .03])
            qd[:] = torch.linspace(-.4, .5, model.joint_dof_count)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            rows.append(TensorDict({'robot':{
                'joint_pos':q[q_indices].clone(),
                'joint_vel':qd[[builder.joint_qd_start[j] for j in (hinge, slide)]].clone(),
                'body_pose_w':wp.to_torch(state.body_q).clone(),
                'body_twist_w':wp.to_torch(state.body_qd).clone(),
            }}, batch_size=[]))
        self.physical = torch.stack(rows)

    def test_roundtrip_newton_poses_and_com_velocities(self):
        state = self.representation.encode(self.physical)
        decoded = self.representation.decode(state)
        for key in self.physical.keys(True, True):
            torch.testing.assert_close(decoded[key], self.physical[key], atol=2e-6, rtol=2e-6)
        self.assertEqual(state['root_pose'].shape, (3, 7))
        self.assertEqual(self.representation.features(state).shape, (3, 14))
        batched = state.unsqueeze(1).expand(3, 2)
        self.assertEqual(self.representation.decode(batched)['robot', 'body_pose_w'].shape, (3, 2, 4, 7))

    def test_zero_prediction_and_gradients_through_fk(self):
        rep = self.representation.double()
        state = rep.encode(self.physical.to(dtype=torch.float64))
        delta = torch.zeros(3, rep.prediction_dim, dtype=torch.float64, requires_grad=True)
        predicted = rep.apply_prediction(state, delta)
        torch.testing.assert_close(predicted, state)
        loss = rep.decode(predicted)['robot', 'body_pose_w'].square().sum() + rep.decode(predicted)['robot', 'body_twist_w'].square().sum()
        gradient, = torch.autograd.grad(loss, delta)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[..., 12:].abs().sum(), 0)
        def forward(prediction):
            value = rep.decode(rep.apply_prediction(state, prediction))
            return value['robot', 'body_pose_w'], value['robot', 'body_twist_w']
        self.assertTrue(torch.autograd.gradcheck(forward, (delta,), fast_mode=True))

    def test_task_preparation_independent_of_policy_observations(self):
        task = G1MotionTracking()
        task.output_physical_fields = (('robot', 'projected_gravity'),)
        task.output_mdp_fields = ('clip_end',)
        config = SuperMimicConfig(policy_obs_keys=('command',), critic_obs_keys=('proprio',))
        prepared = config.prepare_task(task)
        self.assertEqual(task.output_physical_fields, (('robot', 'projected_gravity'),))
        self.assertEqual(task.output_mdp_fields, ('clip_end',))
        self.assertEqual(set(prepared.output_physical_fields), set(config.representation.physical_fields) | {('robot', 'projected_gravity')})
        self.assertNotIn('reference', prepared.output_mdp_fields)
        self.assertIn('motion_t', prepared.output_mdp_fields)
        prepared.mdp.observations.clear()
        self.assertTrue(task.mdp.observations)
        other = SuperMimicConfig(policy_obs_keys=(), critic_obs_keys=()).prepare_task(task)
        self.assertEqual(other.output_physical_fields, prepared.output_physical_fields)

    def test_build_rejects_missing_outputs_and_unsupported_fields(self):
        from newrl.mdp.base import ObservationGroup
        from newrl.mdp.observations import EntityField

        cfg = SuperMimicConfig(policy_obs_keys=('state',), critic_obs_keys=('state',),
                              actor_hidden=(16,), critic_hidden=(16,), wm_hidden=(16,), wm_observations={})
        mdp = MDP(observations={'state':ObservationGroup({'q':EntityField('robot','joint_pos')})})._initialize(self.scene)
        env = SimpleNamespace(scene=self.scene, mdp=mdp, device=torch.device('cpu'), action_dim=2,
                              config=SimpleNamespace(output_physical_fields=(), terrain='plane'), output_mdp_fields=())
        with self.assertRaisesRegex(ValueError, 'prepare_task'):
            cfg.build(env)
        env.config.output_physical_fields = cfg.representation.physical_fields
        env.output_mdp_fields = env.mdp.replay_fields
        learner = cfg.build(env)
        self.assertIsNot(learner.mdp, env.mdp)
        self.assertEqual(learner.mdp.observations.keys(), env.mdp.observations.keys())
        env.mdp = MDP(observations={'state':SimpleNamespace(terms={'bad':SimpleNamespace(physical_fields={('robot', 'joint_torque')})})})
        with self.assertRaisesRegex(ValueError, 'joint_torque'):
            cfg.build(env)

    def test_trainer_uses_prepared_task_for_env_logger_and_checkpoint(self):
        from newrl.commands import train as module
        from newrl.configs import EnvironmentConfig

        policy = Mock()
        policy.train_op.return_value = {}
        original = EnvironmentConfig(name='test-task')
        prepare = Mock(side_effect=SuperMimicConfig().prepare_task)
        algorithm = SimpleNamespace(name='test-algo', update_every=2, prepare_task=prepare, build=lambda _: policy)
        cfg = module.TrainConfig(task=original, algo=algorithm, iterations=1,
                                 checkpoint_interval=1, compile_policy=False,
                                 memory_profile_iters=0, discard_unused_obs=False)
        carry, batch = TensorDict({}, [2]), TensorDict({}, [2, 2])
        env = SimpleNamespace(device=torch.device('cpu'), mdp=SimpleNamespace(rewards={}), reset=lambda: carry)
        stats = Mock(); stats.update.return_value = {}
        with ExitStack() as stack:
            construct = stack.enter_context(patch.object(module, 'StatefulEnvironment', return_value=env))
            logger = stack.enter_context(patch.object(module, '_make_logger'))
            save = stack.enter_context(patch.object(module, '_save_checkpoint', return_value=Path('/tmp/test-checkpoint.pt')))
            stack.enter_context(patch.object(module, 'EpisodeStats', return_value=stats))
            stack.enter_context(patch.object(module, 'collect_rollout', return_value=(batch, carry)))
            stack.enter_context(patch.object(module, 'setproctitle'))
            stack.enter_context(redirect_stdout(StringIO()))
            module.train.__wrapped__(cfg)
        prepare.assert_called_once_with(original)
        prepared = construct.call_args.args[0]
        self.assertEqual(prepared.output_physical_fields, SuperMimicConfig().representation.physical_fields)
        self.assertIs(logger.call_args.args[0].task, prepared)
        self.assertIs(save.call_args.args[2].task, prepared)
        self.assertEqual(original.output_physical_fields, ())


if __name__ == '__main__': unittest.main()
