"""Roll out a zero-action policy with optional visualization."""

from dataclasses import dataclass

import newton
import torch

from newrl.env import StatefulEnvironment
from newrl.mdp.episode_ends import Timeout
from newrl.utils.config import main

from newrl.configs import EnvironmentConfig  # noqa: E402


@dataclass
class PlayConfig:
    task: EnvironmentConfig
    episode_steps: int = 1000
    steps: int | None = None
    viewer: bool = True

    def __post_init__(self):
        if self.episode_steps < 1 or (self.steps is not None and self.steps < 1):
            raise ValueError("episode_steps and steps must be positive")


@main(PlayConfig, defaults=("task.num_envs=4",))
def main(cfg: PlayConfig) -> None:
    viewer = newton.viewer.ViewerGL() if cfg.viewer else None
    try:
        cfg.task.mdp.truncations["timeout"] = Timeout(cfg.episode_steps)
        env = StatefulEnvironment(cfg.task)
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        if viewer is not None:
            viewer.set_model(env.model)
            if env.num_envs > 1:
                viewer.set_world_offsets((2.0, 2.0, 0.0))
        env.reset()
        steps = resets = 0
        while (cfg.steps is None or steps < cfg.steps) and (
            viewer is None or viewer.is_running()
        ):
            if viewer is None or not viewer.is_paused():
                transition = env.step(actions)
                steps += 1
                finished = transition["done"].squeeze(-1).nonzero(as_tuple=True)[0]
            else:
                finished = None
            if viewer is not None:
                viewer.begin_frame(steps * env.control_dt)
                viewer.log_state(env.state)
                viewer.end_frame()
            if finished is not None and finished.numel():
                env.reset(finished)
                resets += finished.numel()
        print(f"worlds={env.num_envs} control_steps={steps} world_resets={resets}")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
