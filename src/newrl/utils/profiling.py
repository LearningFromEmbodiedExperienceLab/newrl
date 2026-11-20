"""Opt-in CPU wall time and CUDA stream timing for environment stages."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import torch


class StageProfiler:
    """Collect event pairs without synchronizing on each stage."""

    def __init__(self, device: torch.device):
        self.device = device
        self._samples = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device)
        start.record(stream)
        cpu_start = perf_counter()
        yield
        host_ms = (perf_counter() - cpu_start) * 1000
        end.record(stream)
        self._samples[name].append((start, end, host_ms))

    def totals(self) -> dict[str, tuple[float, float, int]]:
        """Return totals as (stream ms, CPU wall ms, calls) by stage."""
        torch.cuda.synchronize(self.device)
        return {
            name: (
                sum(start.elapsed_time(end) for start, end, _ in samples),
                sum(host_ms for _, _, host_ms in samples),
                len(samples),
            )
            for name, samples in self._samples.items()
        }
