"""CUDA regression tests for checkpoint replay and explicit resets."""

import unittest
from dataclasses import replace

import newton
import torch
import warp as wp
from tensordict import TensorDict

from newrl.configs import (
    AS2Locomotion, AS2WLocomotion, AnymalLocomotion, M20SLocomotion, SpotLocomotion,
)
from newrl.contact import TerrainContactConfig
from newrl.env import StatefulEnvironment, StatelessEnvironment
from newrl.mdp.episode_ends import Timeout
from newrl.solver_adapters import MujocoSimConfig, SteelSimConfig
from newrl.steel import reconstruct_augmented_drive_torque
from newrl.terrain import FlatMeshTerrainConfig, PlaneTerrainConfig


def _reward_term(env, transition, group, term):
    column = env.mdp.rewards[group].keys.index(term)
    return transition["reward", group][..., column : column + 1]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class EnvironmentTests(unittest.TestCase):
    def test_wheel_velocity_control_and_partial_reset(self):
        for factory in (AS2WLocomotion, M20SLocomotion):
            for solver in (MujocoSimConfig(), SteelSimConfig()):
                with self.subTest(robot=factory.__name__, solver=type(solver).__name__):
                    env = StatefulEnvironment(
                        factory(num_envs=2, solver=solver, graph_capture=False)
                    )
                    env.reset()
                    entity = env.scene.entities["robot"]
                    mask = entity.velocity_mask
                    action = torch.zeros(2, env.action_dim, device=env.device)
                    action[:, mask] = 0.5
                    env.step(action)
                    target = wp.to_torch(env.control.joint_target_qd)[entity.qd]
                    torch.testing.assert_close(target[:, mask], torch.full_like(target[:, mask], 5.0))
                    self.assertGreater(
                        float(wp.to_torch(env.state.joint_qd)[entity.qd][:, mask].abs().mean()),
                        0.01,
                    )
                    env.reset(torch.tensor([0], device=env.device))
                    target = wp.to_torch(env.control.joint_target_qd)[entity.qd]
                    self.assertTrue(torch.all(target[0] == 0))
                    self.assertTrue(torch.all(target[1, mask] == 5.0))

    def test_quadruped_flat_step_reflects_observations_and_rewards(self):
        for factory in (
            AnymalLocomotion, SpotLocomotion, AS2Locomotion,
            AS2WLocomotion, M20SLocomotion,
        ):
            with self.subTest(robot=factory.__name__):
                cfg = factory(
                    num_envs=2,
                    contact_observation=True,
                    graph_capture=False,
                    deterministic=True,
                )
                cfg.mdp.joint_position_noise = 0.0
                cfg.mdp.root_xy_noise = 0.0
                cfg.mdp.root_yaw_noise = 0.0
                env = StatefulEnvironment(cfg)
                env.reset()
                env.mdp_state["velocity_command"][:] = torch.tensor(
                    [[0.35, 0.2, -0.4], [0.35, -0.2, 0.4]], device=env.device
                )
                env.mdp_state["command_steps_left"][:] = 2
                action = torch.linspace(-0.2, 0.15, env.action_dim, device=env.device)
                mirror_action = env.scene.action_symmetry_transform().to(env.device)
                transition = env.step(torch.stack((action, mirror_action(action))))

                for name, values in transition["observation"].items():
                    with self.subTest(observation=name):
                        mirror = env.observation_symmetry_transform((name,)).to(
                            env.device
                        )
                        tolerance = 5e-3 if factory is M20SLocomotion else 2e-3
                        torch.testing.assert_close(
                            mirror(values[0]), values[1], atol=tolerance, rtol=2e-3
                        )
                for name, values in transition["reward"].items():
                    with self.subTest(reward=name):
                        torch.testing.assert_close(
                            values[0], values[1], atol=2e-3, rtol=2e-3
                        )
                for name in ("terminated", "truncated", "done"):
                    self.assertTrue(torch.equal(transition[name][0], transition[name][1]))

    def test_contact_history_checkpoint_and_partial_reset(self):
        for solver in (MujocoSimConfig(), SteelSimConfig()):
            with self.subTest(solver=type(solver).__name__):
                cfg = AnymalLocomotion(
                    num_envs=2,
                    solver=solver,
                    graph_capture=True,
                    substeps=3,
                    deterministic=True,
                    output_physical_fields=tuple(
                        ("feet_ground", field)
                        for field in (
                            "first_contact",
                            "current_air_time",
                            "last_air_time",
                            "last_contact_pos_w",
                            "last_contact_pos_valid",
                        )
                    ),
                )
                cfg.contact_sensors["feet_ground"] = TerrainContactConfig(
                    "robot",
                    r"(LF|LH|RF|RH)_FOOT",
                    track_air_time=True,
                    track_first_contact=True,
                    track_contact_position=True,
                )
                env = StatelessEnvironment(cfg)
                stateful = StatefulEnvironment(cfg)
                sim, mdp, initial = env.reset()
                torch.testing.assert_close(
                    stateful.reset(), initial, atol=2e-3, rtol=2e-3
                )
                self.assertFalse(
                    initial[
                        "physical_state", "feet_ground", "last_contact_pos_valid"
                    ].any()
                )
                inputs = TensorDict(
                    {"action": torch.zeros((2, env.action_dim), device=env.device)},
                    batch_size=[2],
                )
                seen = torch.zeros((2, 4), dtype=torch.bool, device=env.device)
                for _ in range(30):
                    sim, mdp, transition = env.step(sim, mdp, inputs)
                    actual = stateful.step(inputs["action"])
                    torch.testing.assert_close(actual, transition, atol=2e-3, rtol=2e-3)
                    seen |= transition["physical_state", "feet_ground", "first_contact"]
                self.assertTrue(seen.any(dim=-1).all())
                sensor = transition["physical_state", "feet_ground"]
                self.assertTrue(sensor["last_contact_pos_valid"].any(dim=-1).all())
                self.assertTrue(torch.isfinite(sensor["last_contact_pos_w"]).all())
                air = sensor["current_air_time"]
                weight = env.mdp.rewards["regularization"].terms["feet_air_time"].weight
                expected_reward = weight * ((air > 0.05) & (air < 0.5)).sum(
                    -1, keepdim=True
                )
                torch.testing.assert_close(
                    _reward_term(env, transition, "regularization", "feet_air_time"),
                    expected_reward.float(),
                )
                next_sim, next_mdp, expected = env.step(sim, mdp, inputs)
                env.step(next_sim, next_mdp, inputs)
                _, _, replay = env.step(sim, mdp, inputs)
                torch.testing.assert_close(replay, expected, atol=2e-3, rtol=2e-3)
                reset_sim, _, reset = env.reset(sim, mdp, env_ids=torch.tensor([0]))
                self.assertFalse(
                    reset["physical_state", "feet_ground", "last_contact_pos_valid"][
                        0
                    ].any()
                )
                torch.testing.assert_close(reset_sim[1], sim[1], atol=0, rtol=0)
                torch.testing.assert_close(
                    reset["physical_state", "feet_ground"][1], sensor[1], atol=0, rtol=0
                )

    def test_steel_torque_matches_single_joint_dynamics(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=2.0, inertia=wp.mat33(1.0))
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            target_q=1.0,
            target_ke=100.0,
            target_kd=2.0,
            damping=0.0,
            armature=0.5,
            effort_limit=5.0,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device="cuda:0")
        solver = newton.solvers.SolverSteel(model, friction_anchor_beta=0.0)
        state, next_state = model.state(), model.state()
        wp.to_torch(state.joint_q)[0] = 0.2
        wp.to_torch(state.joint_qd)[0] = 0.5
        control = model.control()
        solver.step(state, next_state, control, None, 0.01)
        indices = wp.array([0], dtype=wp.int32, device=model.device)
        torque = wp.zeros(1, dtype=wp.float32, device=model.device)
        wp.launch(
            reconstruct_augmented_drive_torque,
            dim=1,
            inputs=[
                indices,
                indices,
                indices,
                state.joint_q,
                state.joint_qd,
                next_state.joint_qd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_effort_limit,
                0.01,
                torque,
            ],
            device=model.device,
        )
        acceleration = (
            wp.to_torch(next_state.joint_qd) - wp.to_torch(state.joint_qd)
        ) / 0.01
        torch.testing.assert_close(
            wp.to_torch(torque), 2.5 * acceleration, atol=2e-4, rtol=2e-4
        )

    def test_steel_augmented_torque(self):
        # The drive's explicit component is clamped before its implicit
        # correction; the resulting force can therefore exceed the limit.
        device = "cuda:0"
        arrays = [
            wp.array(values, dtype=wp.float32, device=device)
            for values in (
                [0.2, -0.4],
                [0.5, -0.2],
                [0.1, 0.8],
                [1.0, -1.0],
                [0.0, 0.3],
                [100.0, 10.0],
                [2.0, 1.0],
                [5.0, 0.0],
            )
        ]
        indices = wp.array([0, 1], dtype=wp.int32, device=device)
        output = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(
            reconstruct_augmented_drive_torque,
            dim=2,
            inputs=[indices, indices, indices, *arrays, 0.01, output],
            device=device,
        )
        expected = torch.tensor(
            [
                5.0 - 3.0 * (0.1 - 0.5),
                -(10.0 * (-0.4 + 1.0 + 0.01 * -0.2) + (-0.2 - 0.3)) - 1.1 * (0.8 + 0.2),
            ],
            device=device,
        )
        torch.testing.assert_close(wp.to_torch(output), expected)

    def test_terrain_contact_sensor_and_slip(self):
        for backend, terrain in (
            ("mujoco", PlaneTerrainConfig()),
            ("steel", FlatMeshTerrainConfig(resolution=4)),
        ):
            with self.subTest(backend=backend, terrain=type(terrain).__name__):
                cfg = AnymalLocomotion(
                    num_envs=2,
                    graph_capture=True,
                    substeps=3,
                    deterministic=True,
                    solver=SteelSimConfig()
                    if backend == "steel"
                    else MujocoSimConfig(),
                    terrain=terrain,
                    contact_observation=True,
                    foot_slip_weight=0.5,
                    output_physical_fields=(
                        ("feet_ground", "force_w"),
                        ("feet_ground", "loaded"),
                    ),
                )
                env = StatelessEnvironment(cfg)
                sim, mdp, first = env.reset()
                self.assertFalse(first["physical_state", "feet_ground", "loaded"].any())
                action = torch.zeros(2, env.action_dim, device=env.device)
                inputs = TensorDict({"action": action}, batch_size=[2])
                for _ in range(30):
                    sim, mdp, transition = env.step(sim, mdp, inputs)
                force = transition["physical_state", "feet_ground", "force_w"]
                loaded = transition["physical_state", "feet_ground", "loaded"]
                self.assertEqual(force.shape, (2, 4, 3))
                self.assertTrue(loaded.any(dim=-1).all())
                self.assertTrue((force[..., 2][loaded] > 0).all())
                privileged = transition["observation", "privileged"]
                torch.testing.assert_close(privileged[:, -4:], loaded.float())
                feet_xy = privileged[:, -16:-4].reshape(2, 4, 3)[..., :2]
                torch.testing.assert_close(
                    _reward_term(env, transition, "regularization", "foot_slip"),
                    -0.5 * (feet_xy.square().sum(-1) * loaded).sum(-1, keepdim=True),
                )
                next_sim, next_mdp, expected = env.step(sim, mdp, inputs)
                env.step(next_sim, next_mdp, inputs)
                _, _, replay = env.step(sim, mdp, inputs)
                torch.testing.assert_close(
                    replay["physical_state", "feet_ground", "force_w"],
                    expected["physical_state", "feet_ground", "force_w"],
                    atol=2e-3,
                    rtol=2e-3,
                )
                _, _, reset = env.reset(sim, mdp, env_ids=torch.tensor([0]))
                reset_force = reset["physical_state", "feet_ground", "force_w"]
                torch.testing.assert_close(reset_force[0], torch.zeros_like(force[0]))
                torch.testing.assert_close(reset_force[1], force[1], atol=0, rtol=0)

    def test_scene_projected_gravity(self):
        env = StatefulEnvironment(AnymalLocomotion(num_envs=2))
        env.reset()
        pose = wp.to_torch(env.state.body_q)
        indices = env.scene.entities["robot"].bodies[:, 0]
        pose[indices, 3:7] = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]], device=env.device
        )
        snapshot = env.scene.read(
            env.state, {"robot": ("projected_gravity", "body_pose_w")}
        )
        gravity = snapshot["robot", "projected_gravity"][:, 0]
        torch.testing.assert_close(gravity, gravity.new_tensor([[0, 0, -1], [0, 0, 1]]))
        pose[indices, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        torch.testing.assert_close(gravity[1], gravity.new_tensor([0, 0, 1]))

    def test_actuator_torque_observable_reset_and_replay(self):
        env = StatelessEnvironment(
            AnymalLocomotion(
                num_envs=2,
                graph_capture=True,
                output_physical_fields=(
                    ("robot", "joint_torque"),
                    ("robot", "joint_vel"),
                ),
            )
        )
        sim, mdp, initial = env.reset()
        torque = initial["physical_state", "robot", "joint_torque"]
        torch.testing.assert_close(torque, torch.zeros_like(torque))
        inputs = TensorDict(
            {"action": torch.full((2, env.action_dim), 0.3, device=env.device)},
            batch_size=[2],
        )
        sim, mdp, transition = env.step(sim, mdp, inputs)
        entity = env.scene.entities["robot"]
        torque = transition["physical_state", "robot", "joint_torque"]
        torch.testing.assert_close(
            torque, wp.to_torch(env.observables.qfrc_actuator)[entity.qd]
        )
        velocity = transition["physical_state", "robot", "joint_vel"]
        energy = _reward_term(env, transition, "regularization", "joint_energy1")
        torch.testing.assert_close(
            energy, -0.01 * (torque * velocity).abs().sum(-1, keepdim=True)
        )
        self.assertGreater(torque.abs().max().item(), 0)
        first_sim, first_mdp, first = env.step(sim, mdp, inputs)
        env.step(first_sim, first_mdp, inputs)
        replay_sim, _, replay = env.step(sim, mdp, inputs)
        torch.testing.assert_close(
            replay["physical_state", "robot", "joint_torque"],
            first["physical_state", "robot", "joint_torque"],
            atol=2e-4,
            rtol=2e-4,
        )
        reset_sim, _, reset = env.reset(
            replay_sim, first_mdp, env_ids=torch.tensor([0])
        )
        torch.testing.assert_close(
            reset["physical_state", "robot", "joint_torque"][0],
            torch.zeros_like(torque[0]),
        )
        torch.testing.assert_close(reset_sim[1], replay_sim[1], atol=0, rtol=0)

    def test_mesh_terrain_spawn_replay_and_partial_reset(self):
        anchors = ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        cfg = AnymalLocomotion(
            num_envs=4,
            terrain=FlatMeshTerrainConfig(resolution=4, spawn_locations=anchors),
            output_physical_fields=(("robot", "body_pose_w"),),
        )
        cfg.mdp = replace(cfg.mdp, root_xy_noise=0.0)
        env = StatelessEnvironment(cfg)
        sim_state, mdp_state, transition = env.reset()
        root_x = transition["physical_state", "robot", "body_pose_w"][:, 0, 0]
        self.assertTrue(torch.isin(root_x, root_x.new_tensor([-2.0, 2.0])).all())
        self.assertIn("ground_mesh", env.model.shape_label)
        repeat_sim, repeat_mdp, _ = env.reset()
        torch.testing.assert_close(repeat_sim, sim_state, atol=0, rtol=0)
        torch.testing.assert_close(repeat_mdp, mdp_state, atol=0, rtol=0)

        action = torch.zeros((4, env.action_dim), device=env.device)
        sim_state, mdp_state, transition = env.step(
            sim_state, mdp_state, TensorDict({"action": action}, batch_size=[4])
        )
        self.assertTrue(torch.isfinite(transition["observation", "proprio"]).all())
        reset_sim, reset_mdp, _ = env.reset(
            sim_state, mdp_state, env_ids=torch.tensor([0, 2])
        )
        torch.testing.assert_close(reset_sim[1::2], sim_state[1::2], atol=0, rtol=0)
        torch.testing.assert_close(reset_mdp[1::2], mdp_state[1::2], atol=0, rtol=0)

    def test_stateful_stateless_rollout_parity(self):
        for graph_capture, substeps in ((False, 4), (True, 4), (True, 3)):
            with self.subTest(graph_capture=graph_capture, substeps=substeps):
                self._check_rollout_parity(graph_capture, substeps)

    def test_steel_rollout_parity(self):
        for graph_capture, substeps in ((False, 4), (True, 4), (True, 3)):
            with self.subTest(graph_capture=graph_capture, substeps=substeps):
                self._check_rollout_parity(graph_capture, substeps, "steel")

    def _check_rollout_parity(self, graph_capture, substeps, solver_backend="mujoco"):
        cfg = AnymalLocomotion(
            num_envs=3,
            graph_capture=graph_capture,
            substeps=substeps,
            deterministic=True,
            solver=SteelSimConfig() if solver_backend == "steel" else MujocoSimConfig(),
        )
        cfg.mdp = replace(cfg.mdp, hold_steps=4)
        cfg.mdp.truncations["timeout"] = Timeout(18)
        stateful = StatefulEnvironment(cfg)
        stateless = StatelessEnvironment(cfg)

        def compare(sim, mdp, actual, expected):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(stateful.mdp_state, mdp, atol=2e-4, rtol=2e-4)
            # Inspect all checkpoint fields without changing either environment.

            for name, value in sim.items():
                source = getattr(stateful.state, name, None)
                if source is None:
                    if name == "qfrc_actuator":
                        source = stateful.observables.qfrc_actuator
                    elif name == "constraint_overflow":
                        torch.testing.assert_close(
                            wp.to_torch(stateful.sim.constraint_overflow)[
                                :3
                            ].reshape_as(value),
                            value,
                        )
                        continue
                    else:
                        source = getattr(stateful.sim.mjw_data, name)
                torch.testing.assert_close(
                    wp.to_torch(source).reshape_as(value),
                    value,
                    atol=2e-4,
                    rtol=2e-4,
                    msg=lambda msg, field=name: f"{field}: {msg}",
                )

        sim, mdp, expected = stateless.reset()
        compare(sim, mdp, stateful.reset(), expected)
        for name, group in stateful.mdp.observations.items():
            self.assertIsInstance(group.shape, torch.Size)
            self.assertEqual(expected["observation", name].shape[1:], group.shape)
        phase = torch.arange(3 * stateful.action_dim, device=stateful.device).reshape(
            3, stateful.action_dim
        )
        for step in range(40):
            action = 0.1 * (phase + 0.17 * step).sin()
            actual = stateful.step(action)
            sim, mdp, expected = stateless.step(
                sim, mdp, TensorDict({"action": action}, batch_size=[3])
            )
            compare(sim, mdp, actual, expected)
            for key in ("terminated", "truncated", "done"):
                self.assertEqual(actual[key].shape, torch.Size([3, 1]))
            for name, group in stateful.mdp.rewards.items():
                self.assertEqual(actual["reward", name].shape, (3, len(group.keys)))
            if step in (17, 27):
                self.assertEqual(
                    actual["truncated"].squeeze(-1).tolist(),
                    [True, False, True] if step == 17 else [False, True, False],
                )
            if step in (9, 17, 27, 35):
                ids = torch.tensor(
                    [0, 2] if step in (17, 35) else [1], device=stateful.device
                )
                sim, mdp, expected = stateless.reset(sim, mdp, env_ids=ids)
                compare(sim, mdp, stateful.reset(ids), expected)

    def test_checkpoint_replay_after_another_branch(self):
        for backend in ("mujoco", "steel"):
            with self.subTest(solver_backend=backend):
                self._check_checkpoint_replay_after_another_branch(backend)

    def _check_checkpoint_replay_after_another_branch(self, backend):
        cfg = AnymalLocomotion(
            num_envs=2,
            substeps=3,
            graph_capture=True,
            solver=SteelSimConfig() if backend == "steel" else MujocoSimConfig(),
            output_physical_fields=(("robot", "joint_pos"),),
            output_mdp_fields=("velocity_command",),
        )
        cfg.mdp = replace(cfg.mdp, hold_steps=25)
        env = StatelessEnvironment(cfg)
        sim, mdp, initial = env.reset()
        nominal = env.scene.entities["robot"].default_position
        displacement = initial["physical_state", "robot", "joint_pos"] - nominal
        self.assertTrue((displacement.abs() > 0).any())
        self.assertTrue(
            (displacement.abs() <= cfg.mdp.joint_position_noise + 1e-6).all()
        )
        replay_sim, replay_mdp, _ = env.reset()
        torch.testing.assert_close(replay_sim, sim, atol=0, rtol=0)
        torch.testing.assert_close(replay_mdp, mdp, atol=0, rtol=0)
        action = torch.full((2, env.action_dim), 0.1, device=env.device)
        inputs = TensorDict({"action": action}, batch_size=[2])
        # Advance through contact so the checkpoint includes solver warm-start state.
        for _ in range(24):
            sim, mdp, _ = env.step(sim, mdp, inputs)
        saved_sim, saved_mdp = sim.clone(), mdp.clone()

        next_sim, next_mdp, expected = env.step(sim, mdp, inputs)
        self.assertFalse(expected["done"].any())
        self.assertEqual(expected["mdp_state", "episode_step"].tolist(), [[25], [25]])
        torch.testing.assert_close(
            expected["velocity_command"], mdp["velocity_command"]
        )
        self.assertFalse(
            torch.equal(next_mdp["velocity_command"], mdp["velocity_command"])
        )
        torch.testing.assert_close(
            expected["observation", "command"],
            next_mdp["velocity_command"],
        )
        torch.testing.assert_close(
            expected["mdp_state", "velocity_command"], next_mdp["velocity_command"]
        )
        other = TensorDict({"action": -action}, batch_size=[2])
        env.step(next_sim, next_mdp, other)
        actual_sim, actual_mdp, actual = env.step(sim, mdp, inputs)

        torch.testing.assert_close(sim, saved_sim, atol=0, rtol=0)
        torch.testing.assert_close(mdp, saved_mdp, atol=0, rtol=0)
        torch.testing.assert_close(actual_sim, next_sim, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(actual_mdp, next_mdp)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

        reset_sim, reset_mdp, reset_output = env.reset(
            next_sim, next_mdp, env_ids=torch.tensor([0])
        )
        self.assertEqual(reset_mdp["episode_step"].tolist(), [[0], [25]])
        self.assertEqual(
            reset_output["mdp_state", "episode_step"].tolist(), [[0], [25]]
        )
        self.assertEqual(expected["mdp_state", "episode_step"].tolist(), [[25], [25]])
        torch.testing.assert_close(reset_sim[1], next_sim[1], atol=0, rtol=0)
        torch.testing.assert_close(reset_mdp[1], next_mdp[1], atol=0, rtol=0)
        self.assertNotEqual(
            reset_mdp["reset_rng"][0].item(), next_mdp["reset_rng"][0].item()
        )

    def test_stateful_step_returns_terminal_successor_without_reset(self):
        cfg = AnymalLocomotion(num_envs=2)
        cfg.output_physical_fields = (("robot", "joint_pos"),)
        cfg.output_mdp_fields = ("velocity_command",)
        cfg.mdp.truncations["timeout"] = Timeout(1)
        env = StatefulEnvironment(cfg)
        initial = env.reset()
        for name in (
            "episode_step",
            "command_rng",
            "reset_rng",
            "command_steps_left",
            "spawn_height",
        ):
            self.assertEqual(env.mdp_state[name].shape, (2, 1))
        self.assertEqual(
            set(initial["mdp_state"].keys()), {"episode_step", "velocity_command"}
        )
        self.assertEqual(initial["mdp_state", "episode_step"].shape, (2, 1))
        action = torch.zeros((2, env.action_dim), device=env.device)
        transition = env.step(action)
        self.assertTrue(transition["done"].all())
        self.assertEqual(env.mdp_state["episode_step"].tolist(), [[1], [1]])
        self.assertEqual(transition["mdp_state", "episode_step"].tolist(), [[1], [1]])

        reset = env.reset(torch.tensor([0]))
        self.assertEqual(env.mdp_state["episode_step"].tolist(), [[0], [1]])
        self.assertEqual(reset["mdp_state", "episode_step"].tolist(), [[0], [1]])
        self.assertEqual(transition["mdp_state", "episode_step"].tolist(), [[1], [1]])
        self.assertEqual(initial["mdp_state", "episode_step"].tolist(), [[0], [0]])
        torch.testing.assert_close(
            reset["physical_state"][1], transition["physical_state"][1], atol=0, rtol=0
        )

    def test_physical_read_plan_and_requested_fields(self):
        from newrl.mdp.base import MDP, ObservationGroup
        from newrl.mdp.observations import EntityField

        cfg = AnymalLocomotion(num_envs=2)
        cfg.mdp = MDP(
            observations={
                "policy": ObservationGroup({"joint": EntityField("robot", "joint_pos")})
            }
        )
        env = StatefulEnvironment(cfg)
        self.assertEqual(env.physical_read_plan, {"robot": ("joint_pos",)})
        self.assertNotIn("physical_state", env.reset())
        cfg.output_physical_fields = (("robot", "joint_pos"),)
        env = StatefulEnvironment(cfg)
        first = env.reset()["physical_state"]
        self.assertEqual(set(first["robot"].keys()), {"joint_pos"})
        saved = first.clone()
        action = torch.full((2, env.action_dim), 0.1, device=env.device)
        result = env.step(action)
        self.assertEqual(set(result["physical_state", "robot"].keys()), {"joint_pos"})
        torch.testing.assert_close(first, saved, atol=0, rtol=0)
        cfg.output_physical_fields = (("robot", "body_pose_w"),)
        extra_env = StatefulEnvironment(cfg)
        self.assertEqual(
            set(extra_env.physical_read_plan["robot"]), {"joint_pos", "body_pose_w"}
        )
        self.assertEqual(
            set(extra_env.reset()["physical_state", "robot"].keys()),
            {"body_pose_w"},
        )
        cfg.physical_fields = (("robot", "missing"),)
        with self.assertRaisesRegex(ValueError, "Unknown physical field"):
            StatefulEnvironment(cfg)

    def test_undeclared_custom_term_keeps_complete_snapshot(self):
        from newrl.mdp.base import MDP, RewardGroup

        class Custom:
            range = (0.0, 0.0)

            def __call__(self, physical_state, mdp_state):
                return physical_state["robot", "body_twist_w"][:, 0, :1] * 0

        cfg = AnymalLocomotion(num_envs=1)
        cfg.mdp = MDP(rewards={"custom": RewardGroup({"term": Custom()})})
        env = StatefulEnvironment(cfg)
        self.assertEqual(
            set(env.physical_read_plan["robot"]),
            {
                "joint_pos",
                "joint_vel",
                "body_pose_w",
                "body_twist_w",
                "joint_torque",
                "projected_gravity",
            },
        )
        self.assertNotIn("physical_state", env.reset())
        env.step(torch.zeros((1, env.action_dim), device=env.device))
