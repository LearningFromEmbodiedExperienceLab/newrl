"""Play a trained policy or roll out zero actions with optional visualization."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from socket import socket

import newton
import torch

from newrl.algos.interfaces import LearnerConfig
from newrl.commands import with_used_observations
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.mdp.episode_ends import Timeout
from newrl.projects import load_projects
from newrl.terrains import PlaneTerrainConfig, Terrain
from newrl.utils.config import cs, load
from newrl.utils.config import main as config_main
from newrl.utils.wandb import resolve_checkpoint

CHECKPOINT_CACHE_DIR = Path(".cache/wandb-checkpoints")


def viewer_world_offsets(terrain: Terrain | None, num_envs: int) -> tuple[float, float, float]:
    """Viewing offset applied on top of each world's simulated pose.

    Every world shares one static terrain in world coordinates. ``set_model``
    also spreads worlds by their own size. Either offset draws a robot over a
    different point on that terrain than the one it contacts, so it appears to
    float or sink wherever the ground height changes. An infinite plane is the
    same height everywhere, so multiple worlds can be separated visually.
    """
    if num_envs > 1 and (terrain == "plane" or isinstance(terrain, PlaneTerrainConfig)):
        return (2.0, 2.0, 0.0)
    return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GLViewerConfig:
    def build(self, env: StatefulEnvironment) -> newton.viewer.ViewerGL:
        return newton.viewer.ViewerGL()


@dataclass(frozen=True)
class ViserViewerConfig:
    host: str = "127.0.0.1"
    port: int = 8081

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError("viewer.port must be between 0 and 65535")

    def build(self, env: StatefulEnvironment) -> newton.viewer.ViewerViser:
        port = viewer_port_or_ephemeral(self.host, self.port)
        return newton.viewer.ViewerViser(host=self.host, port=port)


@dataclass(frozen=True)
class BrowserViewerConfig:
    port: int = 8081
    splat_path: str | None = None
    recording_dir: str = "recordings"

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError("viewer.port must be between 0 and 65535")

    def build(self, env: StatefulEnvironment) -> newton.viewer.ViewerNull:
        from newrl.browser_view import BrowserViewer

        return BrowserViewer(
            env.scene, self.port, self.splat_path,
            recording_dir=self.recording_dir, task_name=env.config.name,
        )


cs.store("gl", GLViewerConfig(), group="viewer")
cs.store("viser", ViserViewerConfig(), group="viewer")
cs.store("browser", BrowserViewerConfig(), group="viewer")


@dataclass
class PlayConfig:
    task: EnvironmentConfig | None = None
    episode_steps: int = 1000
    steps: int | None = None
    viewer: GLViewerConfig | ViserViewerConfig | BrowserViewerConfig | None = field(default_factory=GLViewerConfig)
    checkpoint: str | None = None
    num_envs: int | None = None
    deterministic: bool = True
    algo: LearnerConfig | None = field(default=None, metadata={"preset_group": "algo"})
    discard_unused_obs: bool = False

    def __post_init__(self) -> None:
        if self.episode_steps < 1 or (self.steps is not None and self.steps < 1):
            raise ValueError("episode_steps and steps must be positive")
        if self.num_envs is not None and self.num_envs < 1:
            raise ValueError("num_envs must be positive")


def viewer_port_or_ephemeral(host: str, port: int) -> int:
    """Use the requested port when free, otherwise ask the OS for a free one."""
    if port == 0:
        return 0
    try:
        with socket() as probe:
            probe.bind((host, port))
    except OSError:
        print(
            f"Viewer port {port} is unavailable. Requesting an available port.",
            flush=True,
        )
        return 0
    return port


@config_main(PlayConfig, defaults={"task.num_envs": 4})
def play(cfg: PlayConfig) -> None:
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
            task = replace(load(saved_config["task"]), num_envs=4)
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
    if cfg.num_envs is not None:
        task = replace(task, num_envs=cfg.num_envs)
    prepare_task = getattr(algo, 'prepare_task', None)
    if prepare_task is not None:
        task = prepare_task(task)
    if cfg.discard_unused_obs:
        task = with_used_observations(task, algo)

    viewer = None
    try:
        task.mdp.truncations["timeout"] = Timeout(cfg.episode_steps)
        env = StatefulEnvironment(task)
        policy = None
        if algo is not None:
            policy = algo.build(env)
            if weights is not None:
                policy.load_state_dict(weights)
            policy.eval()
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        if cfg.viewer is not None:
            viewer = cfg.viewer.build(env)
        viser_playback = None
        if viewer is not None:
            viewer.set_model(env.model)
            viewer.set_world_offsets(viewer_world_offsets(task.terrain, env.num_envs))
            if isinstance(cfg.viewer, ViserViewerConfig):
                from newrl.viser_view import ViserPlayback

                viser_playback = ViserPlayback(viewer, env.scene)

        carry = env.reset()
        steps = resets = 0
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
