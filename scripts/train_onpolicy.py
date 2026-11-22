"""Train a clipped PPO policy on a NewRL environment."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from time import perf_counter

import torch
from tensordict import TensorDict

from newrl.algos.ppo import PPO, PPOConfig
from newrl.algos.rollout import collect_rollout
from newrl.configs import EnvironmentConfig
from newrl.env import StatefulEnvironment
from newrl.mdp.base import RewardGroup
from newrl.utils.config import dump, main
from newrl.utils.logging import RunLogger, WandbConfig

TRAIN_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs_train"


@dataclass
class TrainConfig:
    task: EnvironmentConfig
    algo: PPOConfig
    seed: int = 0
    iterations: int = 2000
    logger: WandbConfig = field(default_factory=WandbConfig)
    checkpoint_interval: int = 250

    def __post_init__(self):
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")


def _group_width(mdp, groups: tuple[str, ...]) -> int:
    return sum(mdp.observations[name].shape.numel() for name in groups)


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
    directory: Path, policy: PPO, cfg: TrainConfig, iteration: int, frames: int
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"checkpoint_{iteration:06d}.pt"
    temporary = destination.with_suffix(".tmp")
    try:
        torch.save(
            {
                **policy.training_state_dict(),
                "config": dump(cfg),
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

    def __init__(self, reward_groups: dict[str, RewardGroup]):
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
def main(cfg: TrainConfig) -> None:
    run_id = token_hex(4)
    checkpoint_dir = (
        _task_output_dir(cfg.task.name) / "checkpoints" / run_id
        if cfg.checkpoint_interval
        else None
    )
    torch.manual_seed(cfg.seed)
    cfg.task.mdp.seed = cfg.seed
    env = StatefulEnvironment(cfg.task)
    policy = PPO(
        cfg.algo,
        policy_dim=_group_width(env.mdp, cfg.algo.policy_obs_keys),
        critic_dim=_group_width(env.mdp, cfg.algo.critic_obs_keys),
        action_dim=env.action_dim,
    ).to(env.device)
    carry = env.reset()
    episode_stats = EpisodeStats(env.mdp.rewards)
    logger = _make_logger(cfg, run_id=run_id)
    try:
        for iteration in range(cfg.iterations):
            iteration_start = perf_counter()
            batch, carry = collect_rollout(env, policy, cfg.algo.update_every, carry)
            if env.device.type == "cuda":
                torch.cuda.synchronize(env.device)
            rollout_time = perf_counter() - iteration_start
            metrics = episode_stats.update(batch)
            training_start = perf_counter()
            metrics.update(policy.train_op(batch))
            training_end = perf_counter()
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
            if checkpoint_dir is not None and completed % cfg.checkpoint_interval == 0:
                path = _save_checkpoint(checkpoint_dir, policy, cfg, completed, frames)
                print(f"Saved checkpoint {path}", flush=True)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
