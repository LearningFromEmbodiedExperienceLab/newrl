"""Play a trained policy or roll out zero actions with optional visualization."""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import newton
import torch

from newrl.algos.genpo import GenPO, GenPOConfig
from newrl.algos.ppo import PPO, PPOConfig
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
    algo: str | None = None

    def __post_init__(self):
        if self.episode_steps < 1 or (self.steps is not None and self.steps < 1):
            raise ValueError("episode_steps and steps must be positive")
        if self.num_envs is not None and self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.algo is not None and self.algo not in ("ppo", "genpo"):
            raise ValueError("algo must be ppo or genpo")


@config_main(PlayConfig)
def play(cfg: PlayConfig) -> None:
    if cfg.algo is not None and cfg.checkpoint is None:
        raise ValueError("algo requires a checkpoint")
    task = cfg.task
    algo = None
    weights = None
    learner = None
    if cfg.checkpoint is not None:
        path = resolve_checkpoint(cfg.checkpoint, CHECKPOINT_CACHE_DIR)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # Training from the CLI tags the outer config as __main__.TrainConfig.
        # Its nested task and algorithm fields have importable type names.
        fields = checkpoint["config"]["fields"]
        if task is None:
            task = replace(load(fields["task"]), num_envs=4)
        algo = load(fields["algo"])
        if isinstance(algo, GenPOConfig):
            learner, checkpoint_algo = GenPO, "genpo"
        elif isinstance(algo, PPOConfig):
            learner, checkpoint_algo = PPO, "ppo"
        else:
            raise ValueError(f"Unsupported checkpoint algorithm {type(algo).__name__}")
        if cfg.algo is not None and cfg.algo != checkpoint_algo:
            raise ValueError(
                f"Checkpoint contains {checkpoint_algo}, requested {cfg.algo}"
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
        if learner is not None:
            groups = env.mdp.observations
            policy = learner(
                algo,
                policy_dim=sum(groups[name].shape.numel() for name in algo.policy_obs_keys),
                critic_dim=sum(groups[name].shape.numel() for name in algo.critic_obs_keys),
                action_dim=env.action_dim,
            ).to(env.device)
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
