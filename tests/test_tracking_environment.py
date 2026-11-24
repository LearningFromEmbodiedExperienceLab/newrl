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
    def test_velocity_resets_replay_and_zero_defaults(self):
        for solver in (MujocoSimConfig(), SteelSimConfig()):
            with self.subTest(solver=type(solver).__name__):
                config = EnvironmentConfig(num_envs=2, robots={"robot": assetx_g1},
                                           mdp=VelocityResetMDP(), solver=solver, graph_capture=False)
                env = StatelessEnvironment(config)
                sim, mdp, _ = env.reset()
                robot = env.scene.entities["robot"]
                torch.testing.assert_close(wp.to_torch(env.state.joint_qd)[robot.qd], torch.full((2, 29), .2, device=env.device))
                expected = torch.tensor([1., 2., 3., .2, .3, .4], device=env.device).expand(2, -1)
                torch.testing.assert_close(wp.to_torch(env.state.joint_qd)[robot.root_qd], expected)
                # FK must have incorporated root and joint reset velocities.
                pelvis = next(i for i, name in enumerate(robot.body_names) if name.rsplit("/", 1)[-1] == "pelvis")
                torch.testing.assert_close(wp.to_torch(env.state.body_qd)[robot.bodies[:, pelvis]], expected)
                action = TensorDict({"action": torch.zeros(2, 29, device=env.device)}, [2])
                next_sim, next_mdp, transition = env.step(sim, mdp, action)
                env.mdp.velocities = False
                ids = torch.tensor([0], device=env.device)
                partial, partial_mdp, _ = env.reset(next_sim, next_mdp, env_ids=ids)
                for key in next_sim.keys():
                    torch.testing.assert_close(partial[key][1], next_sim[key][1], atol=0, rtol=0)
                self.assertTrue((wp.to_torch(env.state.joint_qd)[robot.qd[0]] == 0).all())
                self.assertTrue((wp.to_torch(env.state.joint_qd)[robot.root_qd[0]] == 0).all())
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
                transition = live.step(action)
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
