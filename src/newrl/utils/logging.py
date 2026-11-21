"""Optional Weights & Biases logging for a training run."""

from dataclasses import dataclass

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
    """Log scalar metrics when enabled. A disabled logger does not contact wandb."""

    def __init__(self, cfg: WandbConfig, config: object):
        self._run = None
        if not cfg.enabled or cfg.mode == "disabled":
            return
        import wandb

        self._run = wandb.init(
            project=cfg.project,
            name=cfg.name,
            group=cfg.group,
            config=dump(config),
            mode=cfg.mode,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
