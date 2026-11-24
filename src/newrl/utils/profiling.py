"""Opt-in CPU wall time and CUDA stream timing."""

from collections.abc import Iterator

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import torch


class StageProfiler:
    """Collect event pairs without synchronizing on each stage."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._samples = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = end = None
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
        cpu_start = perf_counter()
        yield
        host_ms = (perf_counter() - cpu_start) * 1000
        if end is not None:
            end.record(stream)
        self._samples[name].append((start, end, host_ms))

    def totals(self, *, synchronize: bool = True) -> dict[str, tuple[float, float, int]]:
        """Return (stream ms, CPU wall ms, calls) by stage.

        Pass synchronize=False when all recorded CUDA events have completed.
        """
        if synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            name: (
                sum(start.elapsed_time(end) for start, end, _ in samples if start is not None),
                sum(host_ms for _, _, host_ms in samples),
                len(samples),
            )
            for name, samples in self._samples.items()
        }
