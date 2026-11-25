"""Task and policy setup shared by the playback commands."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from newrl.algos.interfaces import LearnerConfig
from newrl.commands import with_used_observations
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.mdp.episode_ends import Timeout
from newrl.utils.config import load
from newrl.utils.wandb import resolve_checkpoint


CHECKPOINT_CACHE_DIR = Path(".cache/wandb-checkpoints")


@dataclass
class PlaybackConfig:
    task: EnvironmentConfig | None = None
    episode_steps: int = 1000
    checkpoint: str | None = None
    deterministic: bool = True
    algo: LearnerConfig | None = field(default=None, metadata={"preset_group": "algo"})
    discard_unused_obs: bool = False

    def __post_init__(self) -> None:
        if self.episode_steps < 1:
            raise ValueError("episode_steps must be positive")


def build_playback(cfg: PlaybackConfig, *, default_worlds: int, num_envs: int | None = None):
    task = cfg.task
    algo = cfg.algo
    weights = None
    if cfg.checkpoint is not None:
        path = resolve_checkpoint(cfg.checkpoint, CHECKPOINT_CACHE_DIR)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved_config = checkpoint["config"]
        version = saved_config.get("version")
        if version is None:
            # Legacy checkpoints tagged the outer config as __main__.TrainConfig.
            saved_config = saved_config["fields"]
        elif version != 1:
            raise ValueError(f"Unsupported checkpoint config version {version!r}")
        if task is None:
            task = replace(load(saved_config["task"]), num_envs=default_worlds)
        algo = load(saved_config["algo"])
        if not isinstance(algo, LearnerConfig):
            raise ValueError(f"Unsupported checkpoint algorithm {type(algo).__name__}")
        if cfg.algo is not None and cfg.algo.algo_id != algo.algo_id:
            raise ValueError(
                f"Checkpoint contains {algo.algo_id}, requested {cfg.algo.algo_id}"
            )
        if cfg.checkpoint.startswith("run:"):
            print(f"Loaded {path.name} from {cfg.checkpoint}", flush=True)
        weights = checkpoint["policy"]
        del checkpoint
    if task is None:
        raise ValueError("Specify task=<preset> or checkpoint=<path>")
    if num_envs is not None:
        task = replace(task, num_envs=num_envs)
    prepare_task = getattr(algo, "prepare_task", None)
    if prepare_task is not None:
        task = prepare_task(task)
    if cfg.discard_unused_obs:
        task = with_used_observations(task, algo)

    task.mdp.truncations["timeout"] = Timeout(cfg.episode_steps)
    env = StatefulEnvironment(task)
    policy = None
    if algo is not None:
        policy = algo.build(env)
        if weights is not None:
            policy.load_state_dict(weights)
        policy.eval()
    return env, policy


