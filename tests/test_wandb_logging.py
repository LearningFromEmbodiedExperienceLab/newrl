"""W&B metric axes and profiling payloads without a network connection."""

import unittest
from unittest.mock import call, patch

from newrl.utils.logging import RunLogger, WandbConfig


class WandbLoggingTests(unittest.TestCase):
    def test_axes_and_sampled_performance(self) -> None:
        with patch("wandb.init") as init:
            logger = RunLogger(WandbConfig(), {})
            run = init.return_value
            run.define_metric.assert_has_calls(
                [
                    call("learning/*", step_metric="env_frames", step_sync=False),
                    call("performance/*", step_metric="iteration", step_sync=False),
                    call("diagnostics/*", step_metric="iteration", step_sync=False),
                ]
            )

            logger.log(
                {
                    "env_frames": 128,
                    "train/return": 2.5,
                    "train/reward_terms/task/tracking": 0.6,
                    "actor/kl": 0.01,
                    "performance/rollout_time": 0.02,
                    "performance/training_time": 0.03,
                    "performance/train_steps_per_s": 2560,
                    "memory/rollout/torch_peak_mib": 48.5,
                    "memory/rollout/device_used_mib": 64,
                },
                step=0,
            )
            run.log.assert_called_once_with(
                {
                    "iteration": 1,
                    "env_frames": 128,
                    "learning/return": 2.5,
                    "diagnostics/reward_terms/task/tracking": 0.6,
                    "diagnostics/actor/kl": 0.01,
                    "performance/rollout_ms": 20,
                    "performance/update_ms": 30,
                    "performance/train_steps_per_s": 2560,
                    "performance/rollout_peak_allocated_mib": 48.5,
                    "performance/rollout_device_used_mib": 64,
                },
                step=0,
                commit=True,
            )

            logger.log({"env_frames": 256, "train/return": 3}, step=1)
            logger.finish()
            self.assertEqual(run.log.call_count, 2)
            run.log.assert_called_with(
                {"iteration": 2, "env_frames": 256, "learning/return": 3},
                step=1,
                commit=True,
            )

    def test_finish_flushes_profile_details_as_table_rows(self) -> None:
        with patch("wandb.init") as init:
            logger = RunLogger(WandbConfig(), {})
            run = init.return_value
            logger.log(
                {
                    "env_frames": 128,
                    "profile/ppo/update/host_ms": 7.5,
                    "profile/ppo/update/calls": 4,
                    "memory/buffers/optimizer_mib": 32,
                    "performance/world_model_time": 0.04,
                },
                step=0,
            )
            run.log.assert_called_once_with({"iteration": 1, "env_frames": 128}, step=0, commit=True)
            logger.finish()

            table = run.log.call_args.args[0]["diagnostics/profile"]
            self.assertEqual(table.columns, ["iteration", "env_frames", "metric", "value", "unit"])
            self.assertEqual(
                table.data,
                [
                    [1, 128, "profile/ppo/update/host_ms", 7.5, "ms"],
                    [1, 128, "profile/ppo/update/calls", 4, "count"],
                    [1, 128, "memory/buffers/optimizer_mib", 32, "MiB"],
                    [1, 128, "performance/world_model_time", 40, "ms"],
                ],
            )
