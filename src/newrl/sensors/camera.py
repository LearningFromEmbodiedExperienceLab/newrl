"""Shared lifecycle for cameras sampled after the control step."""

import newton

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FrameCamera(Protocol):
    requires_bvh_refit: bool

    def sample(
        self,
        state: newton.State,
        *,
        reset: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> None: ...
    def checkpoint_views(self) -> dict[str, torch.Tensor]: ...


def camera_refresh_mask(
    remaining: torch.Tensor | None,
    interval: int,
    *,
    reset: bool,
    reset_mask: torch.Tensor | None,
    tick: int = 1,
) -> torch.Tensor | None:
    """Advance per-world capture clocks, or restart clocks selected for reset."""
    if remaining is None:
        return reset_mask
    if reset:
        if reset_mask is None:
            remaining.fill_(interval)
        else:
            remaining.masked_fill_(reset_mask, interval)
        return reset_mask
    remaining.sub_(tick)
    due = remaining <= 0
    remaining.add_(due, alpha=interval)
    return due
