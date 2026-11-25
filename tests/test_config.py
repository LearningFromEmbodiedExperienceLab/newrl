"""CLI resolution and tagged JSON round trips for dataclass configs."""

import json
import tempfile
import unittest
import io
from contextlib import redirect_stderr
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from newrl.algos.interfaces import TrainingConfig
from newrl.commands import train as _module
from newrl.commands.train import TrainConfig
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.robots import newton_anymal_c
from newrl.solver_adapters import MujocoSimConfig, SteelSimConfig
from newrl.utils.config import cs, load_config, main, save_config
from newrl.utils.logging import RunLogger, WandbConfig

@dataclass
class _Widget:
    items: tuple[int, ...] = (1, 2)
    window: slice = slice(0, 2)
    build: Callable = newton_anymal_c


@dataclass
class _Holder:
    widget: _Widget
    seed: int = 0


class ConfigTests(unittest.TestCase):
    def test_term_group_config_preserves_order_and_named_overrides(self):
        from newrl.configs import EnvironmentConfig
        from newrl.mdp.base import MDP, ObservationGroup, RewardGroup
        from newrl.mdp.observations import PreviousAction
        from newrl.mdp.rewards import ActionRate, Survival
        from newrl.utils.config import dump, load, resolve

        terms = {"survival": Survival(1.0), "action_rate": ActionRate(0.1)}
        reward = RewardGroup(terms)
        terms.clear()
        self.assertEqual(reward.keys, ("survival", "action_rate"))
        self.assertIsInstance(reward.terms, tuple)
        saved = dump(reward)
        saved["fields"]["terms"] = dump(dict(reward.terms))
        self.assertEqual(load(saved), reward)
        self.assertEqual(load(dump(reward)), reward)

        task = EnvironmentConfig(mdp=MDP(
            observations={"proprio": ObservationGroup((("action", PreviousAction()),))},
            rewards={"regularization": reward},
        ))
        @dataclass
        class Config:
            task: EnvironmentConfig | None = None

        updated = resolve(
            Config, ["task.mdp.rewards.regularization.terms.action_rate.weight=0.2"],
            defaults={"task": task},
        )
        updated_reward = updated.task.mdp.rewards["regularization"]
        self.assertEqual(updated_reward.keys, reward.keys)
        self.assertEqual(dict(updated_reward.terms)["action_rate"].weight, 0.2)

    def test_unknown_presets_report_the_group_and_available_choices(self):
        from newrl.commands.play import PlayConfig
        from newrl.utils.config import resolve

        for config_type, overrides, group, choice in (
            (PlayConfig, ["task.num_envs=4", "task=NoSuchTask", "algo=ppo"], "task", "AnymalLocoFlat"),
            (TrainConfig, ["task=NoSuchTask", "algo=ppo"], "task", "AnymalLocoFlat"),
            (PlayConfig, ["task=AnymalLocoFlat", "algo=NoSuchAlgo"], "algo", "ppo"),
            (PlayConfig, ["task=AnymalLocoFlat", "task.solver.iterations=7", "task.solver=NoSuchSolver"], "solver", "steel"),
            (PlayConfig, ["task=AnymalLocoFlat", "task.terrain.border=NoSuchBorder", "task.terrain=rough_easy"], "terrain.border", "wall"),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(KeyError) as caught:
                resolve(config_type, overrides)
            self.assertIn(f"Unknown {group} config", str(caught.exception))
            self.assertIn(choice, str(caught.exception))

    def test_cli_unknown_task_prints_usage_without_a_traceback(self):
        from newrl.cli import main as cli

        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            cli(["play", "task=NoSuchTask", "algo=ppo"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("Unknown task config 'NoSuchTask'", stderr.getvalue())
        self.assertIn("AnymalLocoFlat", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertNotIn("NoneType", stderr.getvalue())

    def test_optional_task_still_allows_none_and_checkpoint_playback(self):
        from newrl.commands.play import PlayConfig
        from newrl.utils.config import resolve

        for overrides in (["checkpoint=policy.pt"], ["task=None", "checkpoint=policy.pt"]):
            self.assertIsNone(resolve(PlayConfig, overrides).task)

    def test_command_errors_are_not_wrapped_as_config_errors(self):
        from newrl.utils.config import ConfigError

        @main(TrainConfig)
        def entry(cfg):
            raise ValueError("runtime failure")

        with self.assertRaisesRegex(ValueError, "runtime failure") as caught:
            entry(["task=AnymalLocoFlat", "algo=ppo"])
        self.assertNotIsInstance(caught.exception, ConfigError)

    def test_collision_mode_is_independent_of_solver(self):
        from newrl.utils.config import resolve

        for mode in ("newton", "mujoco", "newrl"):
            cfg = resolve(TrainConfig, [
                "task=AnymalLocoFlat", "algo=ppo", f"task.collision={mode}",
            ])
            self.assertEqual(cfg.task.collision, mode)
            self.assertIsInstance(cfg.task.solver, MujocoSimConfig)
        incompatible = resolve(TrainConfig, [
            "task=AnymalLocoFlat", "algo=ppo", "task.collision=mujoco", "task.solver=steel",
        ])
        with self.assertRaisesRegex(ValueError, "requires the MuJoCo solver"):
            StatefulEnvironment(incompatible.task)

    def test_algorithm_presets_resolve_to_concrete_configs(self):
        from newrl.utils.config import resolve

        config = resolve(
            TrainConfig,
            ["task=AnymalLocoFlat", "algo.flow_steps=3", "algo=genpo", "algo.name=trial"],
        )
        self.assertIsInstance(config.algo, TrainingConfig)
        self.assertEqual(config.algo.algo_id, "genpo")
        self.assertEqual(config.algo.flow_steps, 3)
        self.assertEqual(config.algo.name, "trial")
        with self.assertRaisesRegex(KeyError, "Unknown algo config"):
            resolve(TrainConfig, ["task=AnymalLocoFlat", "algo=missing"])

    def test_nested_solver_selection_and_round_trip(self):
        from newrl.utils.config import resolve

        for overrides in (
            [
                "task=AnymalLocoFlat",
                "algo=ppo",
                "task.solver=steel",
                "task.solver.iterations=7",
            ],
            [
                "task.solver.iterations=7",
                "task.solver=steel",
                "algo=ppo",
                "task=AnymalLocoFlat",
            ],
        ):
            config = resolve(TrainConfig, overrides)
            self.assertIsInstance(config.task.solver, SteelSimConfig)
            self.assertEqual(config.task.solver.iterations, 7)
            self.assertFalse(hasattr(config.task.solver, "jacobian"))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                save_config(path, config.task)
                loaded = load_config(path)
            self.assertEqual(loaded.solver, config.task.solver)
        default = resolve(TrainConfig, ["task=AnymalLocoFlat", "algo=ppo"])
        self.assertIsInstance(default.task.solver, MujocoSimConfig)
        self.assertFalse(hasattr(default.task.solver, "articulation_constraint_max"))
        with self.assertRaises((TypeError, ValueError, KeyError)):
            resolve(
                TrainConfig,
                ["task=AnymalLocoFlat", "algo=ppo", "task.solver=missing"],
            )

    @classmethod
    def setUpClass(cls):
        if ("widget", "base") not in cs:
            cs.store("base", _Widget(), group="widget")

    def test_overrides_round_trip_slice_tuple_and_callable(self):
        from newrl.utils.config import resolve

        config = resolve(
            _Holder,
            [
                "widget=base",
                "widget.window=slice(1,4)",
                "widget.items=(4,5,6)",
                "seed=3",
            ],
        )
        self.assertEqual(config.widget.window, slice(1, 4))
        self.assertEqual(config.widget.items, (4, 5, 6))
        self.assertIs(config.widget.build, newton_anymal_c)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(path, config)
            loaded = load_config(path)
        self.assertEqual(loaded, config)

    def test_main_receives_resolved_train_config(self):
        seen = {}

        @main(TrainConfig)
        def entry(cfg: TrainConfig):
            seen["cfg"] = cfg
            return cfg

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            cfg = entry(
                [
                    "task=AnymalLocoFlat",
                    "algo=ppo",
                    "algo.lr=1e-4",
                    "task.num_envs=64",
                    "algo.actor_hidden=(32,32)",
                    "seed=3",
                    f"config_out={path}",
                ]
            )
            saved = path.read_text()
        self.assertIs(seen["cfg"], cfg)
        self.assertEqual(cfg.seed, 3)
        self.assertEqual(cfg.algo.lr, 1e-4)
        self.assertEqual(cfg.algo.actor_hidden, (32, 32))
        self.assertIsInstance(cfg.task, EnvironmentConfig)
        self.assertEqual(cfg.task.num_envs, 64)
        self.assertIn("newrl.configs.EnvironmentConfig", saved)
        self.assertIn("newrl.algos.ppo.PPOConfig", saved)

    def test_missing_group_fails_before_the_function(self):
        called = False

        @main(TrainConfig)
        def entry(cfg: TrainConfig):
            nonlocal called
            called = True

        with self.assertRaises(ValueError) as caught:
            entry(["algo=ppo"])
        self.assertIn("SpotLocoFlat", str(caught.exception))
        self.assertIn("AS2LocoFlat", str(caught.exception))
        self.assertFalse(called)

    def test_default_override_yields_to_the_command_line(self):
        @main(TrainConfig, defaults={"task.num_envs": 4})
        def entry(cfg: TrainConfig):
            return cfg

        self.assertEqual(
            entry(["task=AnymalLocoFlat", "algo=ppo"]).task.num_envs, 4
        )
        self.assertEqual(
            entry(
                ["task=AnymalLocoFlat", "algo=ppo", "task.num_envs=2"]
            ).task.num_envs,
            2,
        )

    def test_structured_defaults_follow_nested_preset_selection(self):
        from newrl.commands.play import PlayConfig
        from newrl.utils.config import resolve

        defaults = {"task.solver.iterations": 7, "task.num_envs": 4}
        for overrides, iterations in (
            (["task.solver=steel", "task=AnymalLocoFlat"], 7),
            (["task.solver.iterations=9", "task.solver=steel", "task=AnymalLocoFlat"], 9),
        ):
            cfg = resolve(PlayConfig, overrides, defaults=defaults)
            self.assertIsInstance(cfg.task.solver, SteelSimConfig)
            self.assertEqual(cfg.task.solver.iterations, iterations)
        self.assertEqual(defaults, {"task.solver.iterations": 7, "task.num_envs": 4})

    def test_play_defaults_skip_absent_task_and_save_resolved_values(self):
        from newrl.commands.play import PlayConfig

        @main(PlayConfig, defaults={"task.num_envs": 4})
        def entry(cfg):
            return cfg

        self.assertIsNone(entry(["checkpoint=policy.pt"]).task)
        self.assertIsNone(entry(["task=None", "checkpoint=policy.pt"]).task)
        self.assertIsNone(entry([]).task)
        for overrides, expected in (
            (["task=AnymalLocoFlat"], 4),
            (["task.num_envs=2", "task=AnymalLocoFlat"], 2),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                cfg = entry([*overrides, f"config_out={path}"])
                self.assertEqual(cfg.task.num_envs, expected)
                self.assertEqual(load_config(path).task.num_envs, expected)

    def test_play_main_passes_overrides_without_rewriting(self):
        from newrl.commands import play as module

        overrides = ["task=AnymalLocoFlat", "task.num_envs=2"]
        with patch.object(module, "load_projects"), patch.object(module, "play") as play:
            module.main(overrides)
            play.assert_called_once_with(overrides)
        self.assertEqual(overrides, ["task=AnymalLocoFlat", "task.num_envs=2"])

    def test_disabled_logger_does_not_open_a_run(self):
        with patch("wandb.init") as init:
            logger = RunLogger(WandbConfig(enabled=False), object())
            logger.log({"actor/policy_loss": 1.0}, step=0)
            logger.finish()
        init.assert_not_called()

    def test_training_logger_groups_runs_by_task(self):
        from newrl.utils.config import resolve

        cfg = resolve(
            TrainConfig,
            ["task=SpotLocoFlat", "algo=ppo", "algo.name=ppo-muon", "logger.enabled=true"],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs_train"
            with patch.object(_module, "TRAIN_OUTPUT_DIR", output), patch("wandb.init") as init:
                logger = _module._make_logger(cfg)
                logger.finish()
                kwargs = init.call_args.kwargs
                self.assertEqual(kwargs["group"], "SpotLocoFlat")
                logged = kwargs["config"]
                self.assertEqual(logged["task"], "SpotLocoFlat")
                self.assertEqual(logged["robot"], "spot")
                self.assertEqual(logged["algo"], "ppo")
                self.assertEqual(logged["algo_config"]["name"], "ppo-muon")
                self.assertEqual(logged["algo_config"]["lr"], cfg.algo.lr)
                tracking = logged["task_config"]["mdp"]["rewards"]["tracking"]["terms"]
                self.assertEqual(tracking["track_xy_velocity"]["weight"], 1.5)
                self.assertNotIn('"__type__"', json.dumps(logged))
                self.assertEqual(kwargs["dir"], output / "SpotLocoFlat")
                self.assertTrue(kwargs["dir"].is_dir())
                self.assertRegex(
                    kwargs["name"], rf"^{kwargs['id']}-SpotLocoFlat-ppo-muon-\d{{8}}-\d{{6}}Z$"
                )

                cfg.logger.name = "custom name"
                cfg.logger.group = "custom group"
                _module._make_logger(cfg).finish()
                self.assertEqual(init.call_args.kwargs["name"], "custom name")
                self.assertEqual(init.call_args.kwargs["group"], "custom group")
                self.assertEqual(init.call_args.kwargs["dir"], output / "SpotLocoFlat")


if __name__ == "__main__":
    unittest.main()
