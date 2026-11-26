"""Play a trained policy or roll out zero actions with optional visualization."""

from dataclasses import dataclass, field

import torch

from newrl.commands.playback import PlaybackConfig, build_playback
from newrl.projects import load_projects
from newrl.utils.config import main as config_main
from newrl.visualizations.configs import BrowserViewerConfig, GLViewerConfig, ViserViewerConfig, viewer_world_offsets
from newrl.visualizations.task import TaskVisualizationPlayback


@dataclass
class PlayConfig(PlaybackConfig):
    steps: int | None = None
    viewer: GLViewerConfig | ViserViewerConfig | BrowserViewerConfig | None = field(default_factory=GLViewerConfig)
    num_envs: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.steps is not None and self.steps < 1:
            raise ValueError("steps must be positive")
        if self.num_envs is not None and self.num_envs < 1:
            raise ValueError("num_envs must be positive")


@config_main(PlayConfig, defaults={"task.num_envs": 4})
def play(cfg: PlayConfig) -> None:
    viewer = None
    try:
        env, policy = build_playback(cfg, default_worlds=4, num_envs=cfg.num_envs)
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        if cfg.viewer is not None:
            viewer = cfg.viewer.build(env)
        viser_playback = None
        task_playback = None
        if viewer is not None:
            viewer.set_model(env.model)
            viewer.set_world_offsets(viewer_world_offsets(env.config.terrain, env.num_envs))
            if isinstance(cfg.viewer, ViserViewerConfig):
                from newrl.visualizations.viser import ViserPlayback

                viser_playback = ViserPlayback(viewer, env.scene)
            elif isinstance(cfg.viewer, BrowserViewerConfig) and cfg.viewer.task_visualizations:
                task_playback = TaskVisualizationPlayback(env, viewer)

        carry = env.reset()
        steps = resets = 0
        marker_step = None
        while (cfg.steps is None or steps < cfg.steps) and (
            viewer is None or viewer.is_running()
        ):
            if viewer is None or viewer.should_step():
                decision = carry.clone()
                if policy is not None:
                    decision = policy.act(decision, deterministic=cfg.deterministic)
                else:
                    decision["action"] = actions
                transition = env.step(decision)
                carry = transition.select(*carry.keys())
                steps += 1
                finished = transition["done"].squeeze(-1).nonzero(as_tuple=True)[0]
            else:
                finished = None
            if viewer is not None:
                viewer.begin_frame(steps * env.control_dt)
                if task_playback is not None and marker_step != steps:
                    task_playback.update()
                    marker_step = steps
                viewer.log_state(env.state)
                if viser_playback is not None:
                    viser_playback.update(env.state)
                viewer.end_frame()
            if finished is not None and finished.numel():
                reset = env.reset(finished)
                carry[finished] = reset.select(*carry.keys())[finished]
                resets += finished.numel()
        print(f"worlds={env.num_envs} control_steps={steps} world_resets={resets}")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


def main(argv: list[str] | None = None) -> None:
    load_projects()
    play(argv)
