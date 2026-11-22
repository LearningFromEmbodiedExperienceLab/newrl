"""CLI resolution and tagged JSON round trips for dataclass configs."""

import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from newrl.algos.interfaces import OnPolicyConfig
from newrl.commands import train as _module
from newrl.commands.train import TrainConfig
from newrl.configs import EnvironmentConfig
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
    def test_algorithm_presets_resolve_to_concrete_configs(self):
        from newrl.utils.config import resolve

        config = resolve(
            TrainConfig,
            ["task=AnymalLocoFlat", "algo.flow_steps=3", "algo=genpo", "algo.name=trial"],
        )
        self.assertIsInstance(config.algo, OnPolicyConfig)
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
        @main(TrainConfig, defaults=("task.num_envs=4",))
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

    def test_disabled_logger_does_not_open_a_run(self):
        with patch("wandb.init") as init:
            logger = RunLogger(WandbConfig(), object())
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
