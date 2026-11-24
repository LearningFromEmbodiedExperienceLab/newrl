"""Checkpoint selection and publication using a mocked W&B service."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from newrl.commands.play import PlayConfig
from newrl.utils.config import resolve
from newrl.utils.logging import RunLogger, WandbConfig
from newrl.utils.wandb import resolve_checkpoint


class WandbCheckpointTests(unittest.TestCase):
    def test_play_resolves_algo_before_checkpoint(self):
        cfg = resolve(PlayConfig, ["algo=ppo", "checkpoint=run:entity/project/run_id"])
        self.assertEqual(cfg.algo.algo_id, "ppo")
        self.assertEqual(cfg.checkpoint, "run:entity/project/run_id")

    def test_latest_is_rechecked_and_pinned_checkpoint_is_cached(self):
        first = Mock(size=3)
        first.name = "checkpoint_000250.pt"
        next_file = Mock(size=3)
        next_file.name = "checkpoint_001000.pt"
        for file in (first, next_file):
            def download(root, name=file.name):
                path = Path(root) / name
                path.write_bytes(b"abc")
                return path.open("rb")

            file.download.side_effect = download
        pending = Mock(size=0)
        pending.name = "checkpoint_002000.pt"

        spec = "run:entity/project/run_id"
        with tempfile.TemporaryDirectory() as directory, patch("wandb.Api") as api:
            cache = Path(directory)
            run = api.return_value.run.return_value
            run.files.return_value = [first]
            cached = resolve_checkpoint(spec, cache)
            self.assertEqual(cached.read_bytes(), b"abc")
            self.assertEqual(cached.parent, cache / "entity" / "project" / "run_id")
            run.files.return_value = [next_file, pending, first]
            self.assertEqual(resolve_checkpoint(spec, cache).name, next_file.name)
            self.assertEqual(resolve_checkpoint(f"{spec}:250", cache), cached)
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint 500"):
                resolve_checkpoint(f"{spec}:500", cache)
            first.download.assert_called_once()
            next_file.download.assert_called_once()
            pending.download.assert_not_called()

    def test_checkpoint_publication_uses_now_policy(self):
        path = Path("run/checkpoint_000250.pt")
        with patch("wandb.init") as init:
            logger = RunLogger(WandbConfig(enabled=True), {})
            logger.save_checkpoint(path)
            logger.finish()
            init.return_value.save.assert_called_once_with(
                path, base_path=path.parent, policy="now", glob=False
            )
            init.reset_mock()
            RunLogger(WandbConfig(enabled=False), {}).save_checkpoint(path)
            init.assert_not_called()
