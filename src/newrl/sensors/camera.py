"""Shared lifecycle for cameras sampled after the control step."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FrameCamera(Protocol):
    requires_bvh_refit: bool

    def sample(self, state, *, reset=False, reset_mask=None) -> None: ...
    def checkpoint_views(self) -> dict[str, torch.Tensor]: ...


def camera_refresh_mask(remaining, interval, *, reset, reset_mask, tick=1):
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
