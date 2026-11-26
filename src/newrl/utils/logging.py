"""Optional Weights & Biases logging for a training run."""

from dataclasses import dataclass
from pathlib import Path
from math import isfinite
import sys

from newrl.utils.config import config_for_logging


_PERFORMANCE_METRICS = {
    "performance/rollout_time": ("performance/rollout_ms", 1000),
    "performance/training_time": ("performance/update_ms", 1000),
    "performance/iter_time": ("performance/iteration_ms", 1000),
    "performance/rollout_fps": ("performance/rollout_steps_per_s", 1),
    "performance/train_steps_per_s": ("performance/train_steps_per_s", 1),
    "memory/rollout/torch_peak_mib": ("performance/rollout_peak_allocated_mib", 1),
    "memory/update/torch_peak_mib": ("performance/update_peak_allocated_mib", 1),
    "memory/rollout/device_used_mib": ("performance/rollout_device_used_mib", 1),
    "memory/update/device_used_mib": ("performance/update_device_used_mib", 1),
}


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "newrl"
    name: str | None = None
    group: str | None = None
    mode: str = "online"
    init_timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("logger.mode must be online, offline, or disabled")
        if not isfinite(self.init_timeout) or self.init_timeout <= 0:
            raise ValueError("logger.init_timeout must be finite and positive")


class RunLogger:
    """Log training metrics and checkpoints to W&B when enabled."""

    def __init__(
        self,
        cfg: WandbConfig,
        config: object,
        *,
        run_id: str | None = None,
        directory: Path | None = None,
    ) -> None:
        self._run = None
        self._profile_table = None
        self._pending_profile_rows = 0
        if not cfg.enabled or cfg.mode == "disabled":
            return
        import wandb

        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        print(
            f"Initializing W&B (mode={cfg.mode}, timeout={cfg.init_timeout:g}s)...",
            flush=True,
        )
        try:
            self._run = wandb.init(
                project=cfg.project,
                id=run_id,
                dir=directory,
                name=cfg.name,
                group=cfg.group,
                config=config_for_logging(config),
                mode=cfg.mode,
                settings=wandb.Settings(init_timeout=cfg.init_timeout),
            )
        except Exception as error:
            print(f"W&B initialization failed: {error}", file=sys.stderr, flush=True)
            raise
        self._run.define_metric("env_frames", hidden=True)
        self._run.define_metric("iteration", hidden=True)
        self._run.define_metric("learning/*", step_metric="env_frames", step_sync=False)
        self._run.define_metric("performance/*", step_metric="iteration", step_sync=False)
        self._run.define_metric("diagnostics/*", step_metric="iteration", step_sync=False)
        print("W&B initialized.", flush=True)

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._run is None:
            return

        iteration = step + 1
        env_frames = metrics["env_frames"]
        row: dict[str, object] = {"iteration": iteration, "env_frames": env_frames}
        profile_rows = []
        for key, value in metrics.items():
            if key == "env_frames":
                continue
            if key in _PERFORMANCE_METRICS:
                name, scale = _PERFORMANCE_METRICS[key]
                row[name] = value * scale
            elif key.startswith("memory/"):
                unit = "MiB" if key.endswith("_mib") else "flag"
                profile_rows.append((iteration, env_frames, key, value, unit))
            elif key.startswith("profile/"):
                unit = "ms" if key.endswith("_ms") else "count"
                profile_rows.append((iteration, env_frames, key, value, unit))
            elif key.startswith("performance/") and key.endswith("_time"):
                profile_rows.append((iteration, env_frames, key, value * 1000, "ms"))
            elif key.startswith("train/") and not key.startswith("train/reward_terms/"):
                row[f"learning/{key.removeprefix('train/')}"] = value
            else:
                row[f"diagnostics/{key.removeprefix('train/')}"] = value

        if profile_rows:
            if self._profile_table is None:
                from wandb import Table

                self._profile_table = Table(
                    columns=["iteration", "env_frames", "metric", "value", "unit"],
                    log_mode="INCREMENTAL",
                )
            for profile_row in profile_rows:
                self._profile_table.add_data(*profile_row)
            self._pending_profile_rows += len(profile_rows)
            if self._pending_profile_rows >= 256:
                row["diagnostics/profile"] = self._profile_table
                self._pending_profile_rows = 0
        self._run.log(row, step=step, commit=True)

    def save_checkpoint(self, path: Path) -> None:
        if self._run is not None:
            self._run.save(path, base_path=path.parent, policy="now", glob=False)

    def finish(self) -> None:
        if self._run is not None:
            if self._pending_profile_rows:
                self._run.log({"diagnostics/profile": self._profile_table})
            self._run.finish()
            self._run = None
            self._profile_table = None
            self._pending_profile_rows = 0
