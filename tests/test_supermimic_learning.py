"""Replay boundaries, supervised model fitting, imagination and checkpoints."""

import copy
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

import test_supermimic as fixtures
from newrl.algos.supermimic import SuperMimicConfig
from newrl.algos.supermimic.replay import ReplayBuffer
from newrl.algos.supermimic.supermimic import RewardNormalization, lambda_returns
from newrl.configs import EnvironmentConfig
from newrl.mdp.base import MDP, ObservationGroup, RewardGroup
from newrl.mdp.observations import EntityField
from newrl.mdp.episode_ends import Timeout
from newrl.mdp.rewards import ActionRate


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        fixtures.RepresentationTests.setUp(self)
        def height(physical, memory):
            return -(physical['robot','body_pose_w'][..., 0, 2:3] - 1.25).square()
        height.physical_fields = frozenset({('robot','body_pose_w')})
        height.range = (-float('inf'),0.)
        self.cfg = SuperMimicConfig(
            wm_observations={}, optim='adamw', critic_optim='adamw', normalize_reward=False,
            policy_obs_keys=('state',), critic_obs_keys=('state',), actor_hidden=(16,), critic_hidden=(16,), wm_hidden=(32,),
            wm_batch=8, wm_window=2, wm_updates=1, warmup_steps=2, buffer_steps=16,
            imagination_batch=8, horizon=3, critic_epochs=1, critic_minibatches=1, wm_lr=.003,
        )
        task = self.cfg.prepare_task(EnvironmentConfig(mdp=MDP(
            observations={'state':ObservationGroup({'q':EntityField('robot','joint_pos'), 'v':EntityField('robot','joint_vel')})},
            rewards={'tracking':RewardGroup({'height':height, 'action':ActionRate(.01)})}, truncations={'timeout':Timeout(4)},
        )))
        mdp = task.mdp._initialize(self.scene)
        self.env = SimpleNamespace(scene=self.scene, mdp=mdp, device=torch.device('cpu'), action_dim=2,
                                   config=task, output_mdp_fields=task.output_mdp_fields)
        self.learner = self.cfg.build(self.env)
        self.learner.replay = ReplayBuffer(16,3,'cpu')
        state = self.learner.representation.encode(self.physical)
        self.initial = state.clone()
        memory = mdp.initial_state(3,2,torch.device('cpu'))
        for _ in range(20):
            action = torch.randn(3,2) * .2
            delta = torch.zeros(3,16)
            delta[:,2] = .002 + .02 * action[:,0]
            delta[:,12:14] = .02 * action
            successor = self.learner.representation.apply_prediction(state,delta)
            physical = self.learner.representation.decode(successor)
            next_memory, transition = mdp.step(physical,memory,action)
            self.learner.wm.update_normalization(state,action)
            obs = mdp.compute_observation(self.learner.representation.decode(state),memory)
            self.learner.actor_norm.update(obs['state'])
            self.learner.critic_norm.update(obs['state'])
            self.learner.replay.add(TensorDict({
                'state':state,'mdp_state':memory,'action':action,
                'next':{'state':successor,'reward':transition['reward','tracking'].sum(-1,keepdim=True),
                        'done':transition['done'],'terminated':transition['terminated'],'truncated':transition['truncated']},
            },[3]))
            state,memory = successor,next_memory
            if transition['done'].all():
                state=self.initial.clone()
                memory,_=mdp.reset(memory,slice(None))

    def test_wrapped_replay_masks_after_boundaries_and_owns_samples(self):
        replay = self.learner.replay
        self.assertEqual(replay.filled,16)
        sample = replay.sample_windows(128,4)
        self.assertTrue(sample['valid'][:,0].all())
        self.assertTrue((~sample['valid']).any())
        done = sample['next','done']
        for t in range(1,4):
            self.assertFalse((sample['valid'][:,t] & done[:,:t].any(1)).any())
        ending = sample['valid'] & done
        self.assertTrue(ending.any())
        before=replay.storage.clone()
        sample['state','joint_pos'].zero_()
        torch.testing.assert_close(before,replay.storage)

    def test_world_model_fits_real_successors_without_changing_actor(self):
        learner=self.learner
        batch=learner.replay.sample_windows(32,2)
        def error():
            with torch.no_grad():
                predicted=learner.wm.rollout(learner.mdp,batch['state'][:,0],batch['mdp_state'][:,0],batch['action'])
                return learner.representation.compute_loss(predicted['state'],batch['next','state'],batch['valid']).item()
        before=error()
        actor=copy.deepcopy(learner.actor.state_dict())
        norm=copy.deepcopy(learner.wm.state_norm.state_dict())
        learner.config.analytic_reward_loss_weight = 0
        for group in learner.opt_wm.param_groups: group['lr'] = .0003
        for _ in range(100): learner.update_world_model()
        self.assertLess(error(),before * .7)
        torch.testing.assert_close(actor,learner.actor.state_dict())
        torch.testing.assert_close(norm,learner.wm.state_norm.state_dict())

    def test_actor_freezes_world_model_and_normalizers_and_checkpoint_restores(self):
        learner=self.learner
        for _ in range(3): learner.update_world_model()
        model=copy.deepcopy(learner.wm.state_dict())
        actor=copy.deepcopy(learner.actor.state_dict())
        norm=copy.deepcopy(learner.actor_norm.state_dict())
        metrics=learner.update_actor_critic()
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        torch.testing.assert_close(model,learner.wm.state_dict())
        torch.testing.assert_close(norm,learner.actor_norm.state_dict())
        self.assertTrue(any(not torch.equal(value,learner.actor.state_dict()[name]) for name,value in actor.items()))
        self.assertTrue(all(parameter.grad is None for parameter in learner.wm.parameters()))
        saved=copy.deepcopy(learner.training_state_dict())
        other=self.cfg.build(self.env)
        other.load_training_state_dict(saved)
        torch.testing.assert_close(learner.state_dict(),other.state_dict())
        self.assertEqual((other.wm_steps,other.policy_steps),(learner.wm_steps,learner.policy_steps))
        self.assertTrue(other.opt_actor.state)
        self.assertIsNone(other.replay)
        obs=learner.mdp.compute_observation(learner.representation.decode(self.initial),learner.mdp.initial_state(3,2,torch.device('cpu')))
        result=other.eval().act(TensorDict({'observation':obs},[3]),deterministic=True)
        torch.testing.assert_close(result['action'],learner.actor_action(obs))

    def test_lambda_returns_bootstrap_truncations_but_not_falls(self):
        reward=torch.tensor([[[2.]],[[2.]]])
        continuation=torch.tensor([[[1.]],[[0.]]])
        values=torch.tensor([[[9.,8.],[4.,6.]],[[9.,8.],[4.,6.]]])
        target=lambda_returns(reward,continuation,values,.5,.95)
        torch.testing.assert_close(target,torch.tensor([[[4.,5.]],[[2.,2.]]]))

    def test_invalid_config_and_reference_horizon_filter(self):
        for change in ({'warmup_steps':1},{'gamma':1.0},{'action_noise':-1.},{'horizon':0}):
            values=dict(wm_window=2,warmup_steps=2);values.update(change)
            with self.assertRaises(ValueError):SuperMimicConfig(**values)
        replay=ReplayBuffer(2,1,'cpu')
        row=TensorDict({'mdp_state':{'motion_id':torch.zeros(1,1,dtype=torch.long),'motion_t':torch.tensor([[8]])}},[1])
        replay.add(row)
        motion=SimpleNamespace(clip_end=torch.tensor([10]),clip_start=torch.tensor([0]))
        with self.assertRaisesRegex(ValueError,'remaining reference'):
            replay.sample_starts(2,horizon=3,motion=motion)
        self.assertEqual(replay.sample_starts(2,horizon=1,motion=motion).batch_size,(2,))

    def test_unused_diagnostic_observations_do_not_constrain_representation(self):
        unused = EntityField('robot', 'joint_torque')._initialize(self.scene)
        self.env.mdp.observations['diagnostic'] = ObservationGroup({'torque':unused})
        learner = self.cfg.build(self.env)
        self.assertNotIn('diagnostic', learner.mdp.observations)
        self.assertIn('diagnostic', self.env.mdp.observations)
        config = copy.deepcopy(self.cfg)
        config.policy_obs_keys = ('state','diagnostic')
        with self.assertRaisesRegex(ValueError,'joint_torque'):
            config.build(self.env)

    def test_soft_continuation_is_differentiable_and_ignores_timeouts(self):
        from newrl.mdp.episode_ends import TrackingDeviation
        from newrl.mdp.observations import BodyHeight

        term = TrackingDeviation(BodyHeight('robot','pelvis'))._initialize(self.scene)
        mdp = MDP(terminations={'height':term}, truncations={'timeout':Timeout(1)})
        memory = mdp.initial_state(1,2,torch.device('cpu'))
        memory['episode_step'][:] = 2
        root = torch.tensor([[0.,0.,1.,0.,0.,0.,1.]])
        memory['reference'] = TensorDict({'root_pose':root},[1])
        pose = root.unsqueeze(-2).clone(); pose[...,2] += .24
        pose.requires_grad_()
        physical = TensorDict({'robot':{'body_pose_w':pose}},[1])
        self.assertFalse(mdp.compute_termination(physical,memory)['height'].any())
        self.assertTrue(mdp.compute_truncation(physical,memory)['timeout'].all())
        survival=mdp.compute_survival(physical,memory)
        self.assertGreater(survival.item(),.5)
        survival.sum().backward()
        self.assertLess(pose.grad[...,2].item(),0)

    def test_checkpoint_task_and_config_round_trip(self):
        from newrl.commands.train import TrainConfig
        from newrl.utils.config import dump,load,resolve

        config=resolve(TrainConfig,['task=G1LAFAN1Tracking','algo=supermimic'])
        self.assertIsInstance(config.algo,SuperMimicConfig)
        self.assertEqual(load(dump(config.algo)),config.algo)

    def test_dynamics_body_positions_are_independent_inputs_with_real_only_statistics(self):
        from newrl.mdp.observations import BodyPosition

        cfg = copy.deepcopy(self.cfg)
        cfg.wm_observations = {'body_pos_b':BodyPosition('robot', '.*')}
        learner = cfg.build(self.env)
        physical = learner.representation.decode(self.initial)
        memory = learner.mdp.initial_state(3,2,torch.device('cpu'))
        features = learner.wm.observations(physical, memory)
        self.assertEqual(features.shape, (3,12))
        # Pelvis is the reference origin; the other body offsets remain nonzero.
        torch.testing.assert_close(features[:,:3], torch.zeros(3,3))
        self.assertGreater(features[:,3:].abs().sum(),0)
        action = torch.zeros(3,2)
        learner.wm.update_normalization(self.initial, action, physical, memory)
        norm = copy.deepcopy(learner.wm.observation_norm.state_dict())
        learner.wm.rollout(learner.mdp,self.initial,memory,action[:,None].expand(3,2,2))
        torch.testing.assert_close(norm,learner.wm.observation_norm.state_dict())
        self.assertEqual(learner.actor[0].layers[0].in_features,4)
        self.assertEqual(learner.wm.network[0].layers[0].in_features,14+2+12)

    def test_muon_partition_updates_and_optimizer_checkpoint(self):
        from newrl.algos.common.modules import MLP
        from newrl.algos.common.optimizer import build_optimizer

        network = torch.nn.Sequential(MLP((3,8,8),torch.nn.SiLU,first_non_muon=True),torch.nn.Linear(8,2))
        network[-1].weight._non_muon = True
        optimizer = build_optimizer(network,lr=.001,kind='muon_adam')
        muon = {id(p) for group in optimizer.muon.param_groups for p in group['params']}
        self.assertEqual(muon,{id(network[0].layers[3].weight)})
        self.assertIsInstance(optimizer.adamw,torch.optim.Adam)
        self.assertEqual(optimizer.muon.param_groups[0]['weight_decay'],.1)
        before = network[0].layers[3].weight.detach().clone()
        network(torch.randn(4,3)).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before,network[0].layers[3].weight))
        other = build_optimizer(network,lr=.001,kind='muon_adam')
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for kind in ('muon','adamw'):
            first,second = other.state_dict()[kind],optimizer.state_dict()[kind]
            self.assertEqual(first['param_groups'],second['param_groups'])
            torch.testing.assert_close(first['state'],second['state'])

    def test_reward_normalization_matches_return_std_and_resets_boundaries(self):
        norm = RewardNormalization(.5,2.)
        reward = torch.tensor([[[1.],[2.],[3.]],[[2.],[1.],[1.]]])
        done = torch.tensor([[[False],[True],[False]],[[False],[False],[False]]])
        norm.update(reward,done)
        returns = torch.tensor([1.,2.5,3.,2.,2.,2.])
        expected = torch.maximum((returns.var(correction=0)+1e-8).sqrt(),returns.abs().max()/4.)
        torch.testing.assert_close(norm.scale(),expected)
        torch.testing.assert_close(norm.returns,torch.tensor([[3.],[2.]]))
        address = norm.returns.data_ptr()
        norm.update(torch.zeros(2,1,1),torch.zeros(2,1,1,dtype=torch.bool))
        self.assertEqual(norm.returns.data_ptr(),address)

    def test_reference_defaults(self):
        cfg = SuperMimicConfig()
        self.assertEqual((cfg.wm_updates,cfg.imagination_batch,cfg.horizon,cfg.critic_epochs),(10,4096,32,5))
        self.assertEqual(cfg.wm_hidden,(2048,2048,2048))
        self.assertEqual(cfg.actor_hidden,(512,512,512))
        self.assertEqual((cfg.optim,cfg.critic_optim,cfg.amp),('muon_adam','muon_adam','bf16'))
        self.assertTrue(cfg.normalize_reward and cfg.compile_world_model and cfg.capture_wm_update and cfg.capture_policy_critic_update)


if __name__=='__main__':unittest.main()
