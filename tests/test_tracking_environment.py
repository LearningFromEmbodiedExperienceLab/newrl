"""CUDA checks for velocity resets and the PPO motion-tracking path."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from newrl.algos.ppo import PPOConfig
from newrl.algos.rollout import collect_rollout
from newrl.configs import EnvironmentConfig, G1MotionTracking
from newrl.datasets import LAFAN1Dataset
from newrl.datasets.lafan1 import JOINT_NAMES
from newrl.env import StatefulEnvironment, StatelessEnvironment
from newrl.mdp.base import MDP
from newrl.mdp.episode_ends import Timeout
from newrl.robots import assetx_g1
from newrl.solver_adapters import MujocoSimConfig, SteelSimConfig


class VelocityResetMDP(MDP):
    velocities = True

    def reset(self, state, ids):
        state, poses = super().reset(state, ids)
        if self.velocities:
            count = len(state["episode_step"][ids])
            root = state["previous_action"].new_tensor([1., 2., 3., .2, .3, .4]).expand(count, -1)
            poses["robot"] = TensorDict({
                "joint_vel": torch.full((count, 29), .2, device=root.device),
                "root_twist": root,
            }, batch_size=[count])
        return state, poses


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TrackingEnvironmentTests(unittest.TestCase):
    def test_supermimic_bf16_compilation_and_capture_replay(self):
        from newrl.algos.supermimic import SuperMimicConfig

        cfg = SuperMimicConfig(
            warmup_steps=2, buffer_steps=16, wm_window=2, wm_batch=8, wm_updates=1,
            imagination_batch=8, horizon=2, actor_hidden=(32,32), critic_hidden=(32,32),
            wm_hidden=(64,64), critic_epochs=1, critic_minibatches=1,
        )
        env = StatefulEnvironment(cfg.prepare_task(G1MotionTracking(num_envs=8,graph_capture=False)))
        learner = cfg.build(env)
        _,carry = learner.collect(env,4,env.reset())
        self.assertEqual(learner.wm.observations.shape, (42,))
        state = learner.representation.encode(carry['physical_state'])
        # Before compilation/capture, verify hidden BF16 and FP32 physical outputs.
        dtype = []
        hooks = [layer.register_forward_hook(lambda module,args,out: dtype.append(out.dtype))
                 for layer in learner.wm.network[0].modules() if isinstance(layer,torch.nn.Linear)]
        with torch.no_grad():
            _,physical = type(learner.wm).predict(learner.wm,state,torch.zeros(8,29,device=env.device),
                                                 carry['physical_state'],carry['mdp_state'])
        for hook in hooks:
            hook.remove()
        self.assertEqual(dtype,[torch.bfloat16,torch.bfloat16])
        self.assertEqual(physical['robot','body_pose_w'].dtype,torch.float32)
        for _ in range(5):
            _,carry = learner.collect(env,2,carry)
            metrics = learner.update_world_model()
            metrics.update(learner.update_actor_critic())
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIsNotNone(learner._wm_update.graph)
        self.assertIsNotNone(learner._policy_update.graph)
        self.assertEqual(learner.wm.observation_norm.count.item(),112)
        self.assertTrue(all(p.grad is None for p in learner.wm.parameters()))
        # Optimizer state reload invalidates captured storage and restarts warmup.
        learner.load_training_state_dict(learner.training_state_dict())
        self.assertIsNone(learner._wm_update.graph)
        self.assertEqual(learner._policy_update.warmup,3)

    def test_supermimic_real_replay_and_separate_updates(self):
        from newrl.algos.supermimic import SuperMimicConfig

        cfg = SuperMimicConfig(
            compile_world_model=False, capture_wm_update=False, capture_policy_critic_update=False,
            amp='none', optim='adamw', critic_optim='adamw',
            warmup_steps=2, buffer_steps=8, wm_window=2, wm_batch=4, wm_updates=1,
            imagination_batch=4, horizon=2, actor_hidden=(16,), critic_hidden=(16,),
            wm_hidden=(32,), critic_epochs=1, critic_minibatches=1,
        )
        task = cfg.prepare_task(G1MotionTracking(num_envs=4, graph_capture=False))
        task.mdp.truncations['timeout'] = Timeout(2)
        env = StatefulEnvironment(task)
        learner = cfg.build(env)
        batch, carry = learner.collect(env, 4, env.reset())
        self.assertEqual(batch.batch_size, (4, 4))
        storage = learner.replay.storage
        self.assertTrue(storage['next', 'done'][1].all())
        self.assertTrue((storage['mdp_state', 'episode_step'][2] == 0).all())
        self.assertGreater((storage['next', 'state', 'root_pose'][1] - storage['state', 'root_pose'][2]).abs().max(), 1e-4)
        self.assertEqual(learner.actor_norm.count.item(), 16)
        before = learner.wm.state_norm.count.clone()
        metrics = learner.update_world_model()
        metrics.update(learner.update_actor_critic())
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        torch.testing.assert_close(before, learner.wm.state_norm.count)
        self.assertTrue(all(parameter.grad is None for parameter in learner.wm.parameters()))
        memory = learner.mdp.restore_replay_state(carry['mdp_state'])
        self.assertTrue((memory['episode_step'] == 0).all())

    def test_supermimic_g1_roundtrip_mdp_equivalence_and_actor_gradients(self):
        from newrl.algos.supermimic import SuperMimicConfig

        config = SuperMimicConfig()
        task = config.prepare_task(G1MotionTracking(num_envs=3, graph_capture=False))
        env = StatefulEnvironment(task)
        learner = config.build(env)
        rep = learner.representation
        env.reset()
        before = env.mdp_state.clone()
        action = torch.full((3, 29), .1, device=env.device)
        real = env.step(TensorDict({"action": action}, action.shape[:-1]))
        physical = real['physical_state']
        compact = rep.encode(physical)
        decoded = rep.decode(compact)
        for key in physical.keys(True, True):
            torch.testing.assert_close(decoded[key], physical[key], atol=2e-4, rtol=2e-4)
        restored = env.mdp.restore_replay_state(real['mdp_state'])
        torch.testing.assert_close(restored, env.mdp_state)
        replay_state, replay = env.mdp.step(decoded, before, action)
        torch.testing.assert_close(replay_state.select(*env.output_mdp_fields), real['mdp_state'])
        for key in ('reward', 'observation', 'terminated', 'truncated', 'done'):
            torch.testing.assert_close(replay[key], real[key], atol=2e-4, rtol=2e-4)
        # A tiny surrogate dynamics map checks the full actor -> prediction -> FK
        # -> analytic MDP path without requiring a fitted world model yet.
        actor = torch.nn.Linear(rep.feature_dim, 29).to(env.device)
        imagined_action = actor(rep.features(compact))
        prediction = torch.cat((torch.zeros(3, 12, device=env.device), .01 * imagined_action, .02 * imagined_action), -1)
        imagined_physical = rep.decode(rep.apply_prediction(compact, prediction))
        _, imagined = env.mdp.step(imagined_physical, before, imagined_action)
        loss = -sum(group.sum() for group in imagined['reward'].values())
        loss.backward()
        self.assertTrue(torch.isfinite(actor.weight.grad).all())
        self.assertGreater(actor.weight.grad.abs().sum(), 0)

    def test_velocity_resets_replay_and_zero_defaults(self):
        for solver in (MujocoSimConfig(), SteelSimConfig()):
            with self.subTest(solver=type(solver).__name__):
                config = EnvironmentConfig(num_envs=2, robots={"robot": assetx_g1},
                                           mdp=VelocityResetMDP(), solver=solver, graph_capture=False)
                env = StatelessEnvironment(config)
                sim, mdp, _ = env.reset()
                robot = env.scene.entities["robot"]
                torch.testing.assert_close(wp.to_torch(env.state.joint_qd)[robot.joint_qd_indices], torch.full((2, 29), .2, device=env.device))
                expected = torch.tensor([1., 2., 3., .2, .3, .4], device=env.device).expand(2, -1)
                torch.testing.assert_close(wp.to_torch(env.state.joint_qd)[robot.root_qd_indices], expected)
                # FK must have incorporated root and joint reset velocities.
                pelvis = next(i for i, name in enumerate(robot.body_names) if name.rsplit("/", 1)[-1] == "pelvis")
                torch.testing.assert_close(wp.to_torch(env.state.body_qd)[robot.body_indices[:, pelvis]], expected)
                action = TensorDict({"action": torch.zeros(2, 29, device=env.device)}, [2])
                next_sim, next_mdp, transition = env.step(sim, mdp, action)
                env.mdp.velocities = False
                ids = torch.tensor([0], device=env.device)
                partial, partial_mdp, _ = env.reset(next_sim, next_mdp, env_ids=ids)
                for key in next_sim.keys():
                    torch.testing.assert_close(partial[key][1], next_sim[key][1], atol=0, rtol=0)
                self.assertTrue((wp.to_torch(env.state.joint_qd)[robot.joint_qd_indices[0]] == 0).all())
                self.assertTrue((wp.to_torch(env.state.joint_qd)[robot.root_qd_indices[0]] == 0).all())
                # A branch/reset must not change replay of the original checkpoint.
                replay_sim, replay_mdp, replay = env.step(sim, mdp, action)
                torch.testing.assert_close(replay_sim, next_sim)
                torch.testing.assert_close(replay_mdp, next_mdp)
                torch.testing.assert_close(replay, transition)

    def test_tracking_stateful_stateless_parity_and_ppo_update(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            robot_config = assetx_g1()
            raw = np.zeros((61, 36))
            raw[:, 2] = .8
            raw[:, 6] = 1
            raw[:, 0] = np.arange(61) / 30 * .1
            # A gentle, nonzero joint velocity exercises reference resets.
            raw[:, 7:] = np.arange(61)[:, None] / 30 * .01
            source = directory / "motion.csv"
            np.savetxt(source, raw, delimiter=",")
            model = robot_config.asset.parent.parent / "model.xml"
            dataset = LAFAN1Dataset(source, model_path=model, cache_dir=directory)
            config = G1MotionTracking(
                num_envs=2, graph_capture=True, substeps=3, sim_dt=0.02 / 3,
                deterministic=True,
            )
            config.mdp.dataset = str(dataset.path)
            config.mdp.truncations["timeout"] = Timeout(2)
            live = StatefulEnvironment(config)
            branch = StatelessEnvironment(config)
            initial = live.reset()
            sim, mdp, initial_branch = branch.reset()
            torch.testing.assert_close(initial, initial_branch)
            initial_snapshot = initial.clone()
            action = torch.zeros(2, 29, device=live.device)
            for _ in range(3):
                transition = live.step(TensorDict({"action": action}, action.shape[:-1]))
                sim, mdp, other = branch.step(sim, mdp, TensorDict({"action": action}, [2]))
                torch.testing.assert_close(transition, other)
            torch.testing.assert_close(initial, initial_snapshot)
            self.assertEqual(transition["reward", "tracking"].shape, (2, 8))
            self.assertEqual(transition["observation", "command"].shape, (2, 389))
            policy = PPOConfig(actor_hidden=(32,), critic_hidden=(32,), num_epochs=1, num_minibatches=1).build(live)
            batch, _ = collect_rollout(live, policy, 4, live.reset())
            metrics = policy.train_op(batch)
            self.assertTrue(metrics)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
