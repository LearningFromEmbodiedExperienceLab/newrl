"""Collectives for synchronous data-parallel training."""

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


@contextmanager
def training_process() -> Iterator[None]:
    """Initialize a torchrun worker, or run normally in a single process."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        yield
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        yield
    finally:
        dist.destroy_process_group()


def reduce_sum(values: torch.Tensor) -> torch.Tensor:
    if world_size() > 1:
        dist.all_reduce(values)
    return values


@torch.no_grad()
def broadcast_module(module: torch.nn.Module) -> None:
    if world_size() > 1:
        for value in (*module.parameters(), *module.buffers()):
            if value.numel():
                dist.broadcast(value, src=0)


@torch.no_grad()
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if world_size() == 1:
        return
    gradients = [p.grad for p in parameters if p.grad is not None]
    flat = torch.cat([grad.reshape(-1) for grad in gradients])
    reduce_sum(flat).div_(world_size())
    offset = 0
    for grad in gradients:
        grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
        offset += grad.numel()


def average_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if world_size() == 1:
        return metrics
    values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
    return dict(zip(metrics, (reduce_sum(values) / world_size()).tolist(), strict=True))


def max_times(times: tuple[float, ...], device: torch.device) -> tuple[float, ...]:
    if world_size() == 1:
        return times
    values = torch.tensor(times, device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return tuple(values.tolist())
