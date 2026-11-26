"""Run a single world in the custom viewer with rewind and branching."""

from dataclasses import dataclass, field
from time import monotonic, sleep

import torch

from newrl.commands.playback import PlaybackConfig, build_playback
from newrl.projects import load_projects
from newrl.visualizations.timeline import Timeline
from newrl.utils.config import main as config_main
from newrl.visualizations.configs import BrowserViewerConfig
from newrl.visualizations.task import TaskVisualizationPlayback


@dataclass
class TimelinePlayConfig(PlaybackConfig):
    viewer: BrowserViewerConfig = field(default_factory=BrowserViewerConfig)
    discard_unused_obs: bool = True
    history_frames: int = 3000
    history_mib: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.viewer, BrowserViewerConfig):
            raise ValueError("Timeline playback requires viewer=browser")
        if self.history_frames < 2 or self.history_mib < 1:
            raise ValueError("Use at least two history frames and a positive memory budget")


@config_main(TimelinePlayConfig, defaults={"task.num_envs": 1})
def play_timeline(cfg: TimelinePlayConfig) -> None:
    env, policy = build_playback(cfg, default_worlds=1, num_envs=1)
    timeline = Timeline(cfg.history_frames, cfg.history_mib * 1024**2, env.control_dt)
    actions = torch.zeros((1, env.action_dim), device=env.device)
    viewer = cfg.viewer.build(env, timeline=timeline)

    def publish(tick, carry, *, jump=False, episode_split=False):
        checkpoint = None
        if carry is not None:
            sim_state, mdp_state = env.snapshot()
            checkpoint = (sim_state.cpu(), mdp_state.cpu(), carry.cpu())
        viewer.begin_frame(tick * env.control_dt, jump=jump)
        if task_playback is not None:
            task_playback.update()
        viewer.log_state(env.state)
        viewer.end_frame(pace=False)
        timeline.append(
            viewer.frame_id, tick, viewer.frame, checkpoint, episode_split=episode_split
        )

    try:
        task_playback = TaskVisualizationPlayback(env, viewer) if cfg.viewer.task_visualizations else None
        viewer.set_model(env.model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        carry = env.reset()
        tick = 0
        publish(tick, carry)
        while viewer.is_running():
            if (branch := timeline.take_branch()) is not None:
                assert branch.checkpoint is not None
                sim_state, mdp_state, carry = (state.to(env.device) for state in branch.checkpoint)
                env.restore(sim_state, mdp_state)
                tick = branch.tick
                publish(tick, carry, jump=True)
                viewer.set_paused(False)
            if not viewer.should_step():
                viewer.camera_previews.sample()
                sleep(0.02)
                continue
            deadline = monotonic() + env.control_dt
            decision = carry.clone()
            if policy is None:
                decision["action"] = actions
            else:
                decision = policy.act(decision, deterministic=cfg.deterministic)
            transition = env.step(decision)
            tick += 1
            finished = transition["done"].squeeze(-1).nonzero(as_tuple=True)[0]
            if finished.numel():
                publish(tick, None)
                carry = env.reset(finished)
                publish(tick, carry, jump=True, episode_split=True)
            else:
                carry = transition.select(*carry.keys())
                publish(tick, carry)
            if (remaining := deadline - monotonic()) > 0:
                sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main(argv: list[str] | None = None) -> None:
    load_projects()
    play_timeline(argv)
