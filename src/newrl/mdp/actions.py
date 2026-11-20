from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointPositionAction:
    """Map normalized actions to offsets from the nominal joint pose."""

    scale: float

    def __call__(self, action: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
        return default + self.scale * action
