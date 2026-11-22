"""Play a trained policy or roll out zero actions with optional visualization."""

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import newton
import torch

from newrl.algos.interfaces import OnPolicyConfig
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.mdp.episode_ends import Timeout
from newrl.utils.config import load, main as config_main
from newrl.utils.wandb import resolve_checkpoint

CHECKPOINT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "wandb-checkpoints"


@dataclass
class PlayConfig:
    task: EnvironmentConfig | None = None
    episode_steps: int = 1000
    steps: int | None = None
    viewer: bool = True
    checkpoint: str | None = None
    num_envs: int | None = None
    deterministic: bool = True
    algo: OnPolicyConfig | None = field(default=None, metadata={"preset_group": "algo"})

    def __post_init__(self):
        if self.episode_steps < 1 or (self.steps is not None and self.steps < 1):
            raise ValueError("episode_steps and steps must be positive")
        if self.num_envs is not None and self.num_envs < 1:
            raise ValueError("num_envs must be positive")


@config_main(PlayConfig)
def play(cfg: PlayConfig) -> None:
    if cfg.algo is not None and cfg.checkpoint is None:
        raise ValueError("algo requires a checkpoint")
    task = cfg.task
    algo = None
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
            task = replace(load(saved_config["task"]), num_envs=4)
        algo = load(saved_config["algo"])
        if not isinstance(algo, OnPolicyConfig):
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
    if cfg.num_envs is not None:
        task = replace(task, num_envs=cfg.num_envs)

    viewer = newton.viewer.ViewerGL() if cfg.viewer else None
    try:
        task.mdp.truncations["timeout"] = Timeout(cfg.episode_steps)
        env = StatefulEnvironment(task)
        policy = None
        if algo is not None:
            policy = algo.build(env)
            policy.load_state_dict(weights)
            policy.eval()
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        if viewer is not None:
            viewer.set_model(env.model)
            if env.num_envs > 1:
                viewer.set_world_offsets((2.0, 2.0, 0.0))
        carry = env.reset().select("observation")
        steps = resets = 0
        while (cfg.steps is None or steps < cfg.steps) and (
            viewer is None or viewer.is_running()
        ):
            if viewer is None or not viewer.is_paused():
                if policy is not None:
                    with policy.policy_norm.freeze():
                        actions = policy.act(carry, deterministic=cfg.deterministic)["action"]
                transition = env.step(actions)
                carry = transition.select("observation")
                steps += 1
                finished = transition["done"].squeeze(-1).nonzero(as_tuple=True)[0]
            else:
                finished = None
            if viewer is not None:
                viewer.begin_frame(steps * env.control_dt)
                viewer.log_state(env.state)
                viewer.end_frame()
            if finished is not None and finished.numel():
                reset = env.reset(finished)
                carry[finished] = reset.select("observation")[finished]
                resets += finished.numel()
        print(f"worlds={env.num_envs} control_steps={steps} world_resets={resets}")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    # Apply the play default only when the CLI selects a task preset.
    if any(token.startswith("task=") for token in argv):
        argv = ["task.num_envs=4", *argv]
    play(argv)


if __name__ == "__main__":
    main()
