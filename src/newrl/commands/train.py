"""Train an on-policy learner on a NewRL environment."""

from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from time import perf_counter

from setproctitle import setproctitle
import torch
from tensordict import TensorDict

from newrl.algos.interfaces import LearnerConfig, Learner
from newrl.algos.rollout import collect_rollout
from newrl.commands import with_used_observations
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.mdp.base import RewardGroup
from newrl.projects import load_projects
from newrl.utils.config import dump, main
from newrl.utils.logging import RunLogger, WandbConfig
from newrl.utils.memory import CudaMemoryProfiler, write_memory_chart

TRAIN_OUTPUT_DIR = Path("outputs_train")


@dataclass
class TrainConfig:
    task: EnvironmentConfig
    algo: LearnerConfig = field(metadata={"preset_group": "algo"})
    seed: int = 0
    iterations: int = 2000
    logger: WandbConfig = field(default_factory=WandbConfig)
    checkpoint_interval: int = 250
    compile_policy: bool = True
    memory_profile_iters: int = 5
    memory_profile_warmup: int = 2
    discard_unused_obs: bool = True

    def __post_init__(self) -> None:
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.memory_profile_iters < 0 or self.memory_profile_warmup < 0:
            raise ValueError("Memory profiling iteration counts must be non-negative")


def _task_output_dir(task: str) -> Path:
    if task in ("", ".", "..") or Path(task).name != task:
        raise ValueError("task.name must be a single directory name")
    return TRAIN_OUTPUT_DIR / task


def _make_logger(cfg: TrainConfig, *, run_id: str | None = None) -> RunLogger:
    if not cfg.logger.enabled or cfg.logger.mode == "disabled":
        return RunLogger(cfg.logger, cfg)

    task = cfg.task.name
    algo = cfg.algo.name
    run_id = token_hex(4) if run_id is None else run_id
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    default_name = f"{run_id}-{task}-{algo}-{timestamp}"
    logger_cfg = replace(
        cfg.logger,
        name=cfg.logger.name if cfg.logger.name is not None else default_name,
        group=cfg.logger.group if cfg.logger.group is not None else task,
    )
    return RunLogger(
        logger_cfg,
        replace(cfg, logger=logger_cfg),
        run_id=run_id,
        directory=_task_output_dir(task),
    )


def _save_checkpoint(
    directory: Path,
    policy: Learner,
    cfg: TrainConfig,
    iteration: int,
    frames: int,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"checkpoint_{iteration:06d}.pt"
    temporary = destination.with_suffix(".tmp")
    try:
        torch.save(
            {
                **policy.training_state_dict(),
                "config": {
                    "version": 1,
                    **{item.name: dump(getattr(cfg, item.name)) for item in fields(cfg)},
                },
                "iteration": iteration,
                "env_frames": frames,
            },
            temporary,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class EpisodeStats:
    """Carry episode returns across rollouts and report causes for completed episodes."""

    def __init__(self, reward_groups: dict[str, RewardGroup]) -> None:
        self.reward_groups = reward_groups
        self.returns: torch.Tensor | None = None
        self.completed_episodes = 0

    def update(self, batch: TensorDict) -> dict[str, float | int]:
        groups, done = batch["reward"], batch["done"]
        names = tuple(groups.keys())
        group_totals = [groups[name].sum(dim=-1, keepdim=True) for name in names]
        reward = (
            torch.cat(group_totals, dim=-1)
            if group_totals
            else done.new_zeros((*batch.batch_size, 0), dtype=torch.float32)
        )
        if self.returns is None:
            self.returns = reward.new_zeros((batch.batch_size[0], len(names)))
        completed_returns = torch.zeros_like(reward)
        for step in range(batch.batch_size[1]):
            self.returns.add_(reward[:, step])
            completed_returns[:, step] = self.returns
            self.returns.masked_fill_(done[:, step], 0)

        completed = int(done.sum())
        self.completed_episodes += completed
        metrics = {
            "train/episodes": self.completed_episodes,
            "train/reward_mean": float(reward.sum(dim=-1).mean()),
        }
        if completed:
            # The batch carries the step count from before the action.
            lengths = batch["mdp_state", "episode_step"][done] + 1
            returns = completed_returns[done.squeeze(-1)]
            metrics.update(
                {
                    "train/episode_len": float(lengths.float().mean()),
                    "train/return": float(returns.sum(dim=-1).mean()),
                    "train/termination_rate": float(
                        batch["terminated"][done].float().mean()
                    ),
                }
            )
            for name, mean in zip(names, returns.mean(dim=0).tolist(), strict=True):
                metrics[f"train/return/{name}"] = mean
            for kind in ("termination", "truncation"):
                terms = batch.get(f"{kind}_terms")
                if terms is not None:
                    for name, flags in terms.items():
                        count = int((flags & done).sum())
                        metrics[f"train/{kind}/{name}"] = count / completed
        for name in names:
            means = groups[name].mean(dim=(0, 1)).tolist()
            for term, mean in zip(self.reward_groups[name].keys, means, strict=True):
                metrics[f"train/reward_terms/{name}/{term}"] = mean
        return metrics


@main(TrainConfig)
def train(cfg: TrainConfig) -> None:
    from newrl.algos.supermimic import SuperMimicConfig

    if isinstance(cfg.algo, SuperMimicConfig):
        from newrl.commands.train_supermimic import train_supermimic

        return train_supermimic(cfg)
    title = f"{cfg.task.name}-{cfg.algo.name}-ETA:"
    setproctitle(f"{title}--:--:--")
    run_id = token_hex(4)
    prepare_task = getattr(cfg.algo, "prepare_task", None)
    if prepare_task is not None:
        # Retain the prepared task in logger/checkpoint config as well as the env.
        cfg = replace(cfg, task=prepare_task(cfg.task))
    checkpoint_dir = (
        _task_output_dir(cfg.task.name) / "checkpoints" / run_id
        if cfg.checkpoint_interval
        else None
    )
    torch.manual_seed(cfg.seed)
    cfg.task.mdp.seed = cfg.seed
    profiler = None
    memory_metrics = {}
    memory_samples = []
    memory_profile_end = min(
        cfg.iterations, cfg.memory_profile_warmup + cfg.memory_profile_iters
    )
    if cfg.memory_profile_iters:
        profiler = CudaMemoryProfiler(torch.device(cfg.task.device))
        profiler.begin_phase()
    task = with_used_observations(cfg.task, cfg.algo) if cfg.discard_unused_obs else cfg.task
    env = StatefulEnvironment(task)
    if profiler is not None:
        memory_metrics.update(profiler.end_phase("env_build"))
    policy = cfg.algo.build(env)
    if cfg.compile_policy:
        policy.act = torch.compile(policy.act, fullgraph=True)
    carry = env.reset()
    episode_stats = EpisodeStats(env.mdp.rewards)
    logger = _make_logger(cfg, run_id=run_id)
    loop_start = perf_counter()
    try:
        for iteration in range(cfg.iterations):
            sample_memory = profiler is not None and (
                iteration == 0
                or cfg.memory_profile_warmup <= iteration
                < cfg.memory_profile_warmup + cfg.memory_profile_iters
            )
            if sample_memory:
                profiler.begin_phase()
            iteration_start = perf_counter()
            batch, carry = collect_rollout(env, policy, cfg.algo.update_every, carry)
            if env.device.type == "cuda":
                torch.cuda.synchronize(env.device)
            rollout_time = perf_counter() - iteration_start
            if sample_memory:
                memory_metrics.update(profiler.end_phase("rollout"))
                memory_metrics.update(profiler.rollout(batch, carry))
            metrics = episode_stats.update(batch)
            if sample_memory:
                profiler.begin_phase()
            training_start = perf_counter()
            metrics.update(policy.train_op(batch))
            training_end = perf_counter()
            if sample_memory:
                memory_metrics.update(profiler.end_phase("update"))
                memory_metrics.update(profiler.environment(env))
                memory_metrics.update(profiler.learner(policy))
                memory_metrics["memory/warmup"] = int(iteration < cfg.memory_profile_warmup)
                metrics.update(memory_metrics)
                memory_samples.append(memory_metrics.copy())
                memory_metrics.clear()
            metrics.update(
                {
                    "performance/rollout_fps": batch.numel() / rollout_time,
                    "performance/rollout_time": rollout_time,
                    "performance/training_time": training_end - training_start,
                    "performance/iter_time": training_end - iteration_start,
                }
            )
            frames = (iteration + 1) * batch.numel()
            logger.log({**metrics, "env_frames": frames}, step=iteration)
            summary = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"iter={iteration} env_frames={frames} {summary}", flush=True)
            completed = iteration + 1
            if memory_samples and completed == memory_profile_end:
                chart_path = (
                    _task_output_dir(cfg.task.name)
                    / "profiles" / run_id / "gpu_memory.svg"
                )
                write_memory_chart(chart_path, memory_samples)
                print(f"Saved GPU memory chart to {chart_path}", flush=True)
            if checkpoint_dir is not None and completed % cfg.checkpoint_interval == 0:
                path = _save_checkpoint(checkpoint_dir, policy, cfg, completed, frames)
                logger.save_checkpoint(path)
                print(f"Saved checkpoint {path}", flush=True)
            remaining = round(
                (perf_counter() - loop_start) / completed * (cfg.iterations - completed)
            )
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            setproctitle(f"{title}{hours:02d}:{minutes:02d}:{seconds:02d}")
    finally:
        logger.finish()


def main(argv: list[str] | None = None) -> None:
    load_projects()
    train(argv)
