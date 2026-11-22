"""Optional Weights & Biases logging for a training run."""

from dataclasses import dataclass
from pathlib import Path

from newrl.utils.config import dump


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "newrl"
    name: str | None = None
    group: str | None = None
    mode: str = "online"

    def __post_init__(self):
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("logger.mode must be online, offline, or disabled")


class RunLogger:
    """Log training metrics and checkpoints to W&B when enabled."""

    def __init__(
        self,
        cfg: WandbConfig,
        config: object,
        *,
        run_id: str | None = None,
        directory: Path | None = None,
    ):
        self._run = None
        if not cfg.enabled or cfg.mode == "disabled":
            return
        import wandb

        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=cfg.project,
            id=run_id,
            dir=directory,
            name=cfg.name,
            group=cfg.group,
            config=dump(config),
            mode=cfg.mode,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def save_checkpoint(self, path: Path) -> None:
        if self._run is not None:
            self._run.save(path, base_path=path.parent, policy="now", glob=False)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
