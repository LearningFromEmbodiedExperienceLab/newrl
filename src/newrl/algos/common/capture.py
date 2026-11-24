"""CUDA learner capture, adapted from revisiting-supermimic's CapturedUpdate."""

from collections.abc import Callable, Sequence
from time import perf_counter

import torch
from tensordict import TensorDict
from torch.optim import Optimizer

from .optimizer import MuonAdamW


class CapturedUpdate:
    """Warm up real updates, then capture/replay with refreshed TensorDict inputs.

    Every call performs exactly one update, including warmup and first capture.
    Parameters, buffers, optimizer storage and hyperparameters must stay stable.
    Returned diagnostics are copied out because replay overwrites graph outputs.
    """

    def __init__(
        self,
        fn: Callable[..., TensorDict],
        module: torch.nn.Module,
        optimizer: Optimizer | MuonAdamW | Sequence[Optimizer | MuonAdamW],
        enabled: bool,
        warmup: int  =  3,
        label: str  =  "WM minibatch",
    ) -> None:
        self.fn, self.module, self.optimizer = fn, module, optimizer
        self.enabled, self.warmup = enabled, warmup
        self.initial_warmup = warmup
        self.label = label
        self.inputs = self.graph = self.outputs = self.identity = self.grads = None
        self.stream = torch.cuda.Stream(device=next(module.parameters()).device) if enabled else None

    def reset(self) -> None:
        """Rewarm after loading optimizer state, whose storage may have changed."""
        self.inputs = self.graph = self.outputs = self.identity = self.grads = None
        self.warmup = self.initial_warmup

    def __call__(self, *inputs: TensorDict) -> TensorDict:
        if not self.enabled:
            return self.fn(*inputs)
        device = next(self.module.parameters()).device
        with torch.cuda.device(device):
            if self.inputs is None:
                self.inputs = tuple(value.detach().clone() for value in inputs)
            else:
                if len(inputs) != len(self.inputs):
                    raise ValueError("Captured update input count changed")
                for destination, source in zip(self.inputs, inputs, strict=True):
                    if destination.batch_size != source.batch_size or set(destination.keys(True, True)) != set(source.keys(True, True)):
                        raise ValueError("Captured update TensorDict structure changed")
                    for key in source.keys(True, True):
                        old, new = destination[key], source[key]
                        if old.shape != new.shape or old.dtype != new.dtype or old.device != new.device:
                            raise ValueError(f"Captured update tensor shape/dtype/device changed: {key}")
                    destination.copy_(source)
            if self.warmup:
                print(f"[capture] {self.label} warmup ({self.warmup} remaining).", flush=True)
                self.stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(self.stream):
                    result = self.fn(*self.inputs)
                torch.cuda.current_stream(device).wait_stream(self.stream)
                self.warmup -= 1
                return result

            if self.graph is None:
                torch.cuda.synchronize(device)
                print(f"[capture] Recording one {self.label} update...", flush=True)
                started = perf_counter()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.outputs = self.fn(*self.inputs)
                self.grads = tuple(parameter.grad for parameter in self.module.parameters())
                print(f"[capture] {self.label} recorded in {perf_counter() - started:.2f}s; replay enabled.", flush=True)

            optimizers = [
                child
                for optimizer in (self.optimizer if isinstance(self.optimizer, (list, tuple)) else [self.optimizer])
                for child in getattr(optimizer, "optimizers", [optimizer])
            ]
            tensors = [*self.module.parameters(), *self.module.buffers()]
            settings = []
            for optimizer in optimizers:
                for state in optimizer.state.values():
                    tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
                for group in optimizer.param_groups:
                    tensors.extend(group["params"])
                    tensors.extend(value for value in group.values() if isinstance(value, torch.Tensor))
                    settings.append(tuple((key, value) for key, value in group.items()
                                          if key != "params" and not isinstance(value, torch.Tensor)))
            identity = (tuple((value.data_ptr(), value.shape, value.dtype, value.device, value.requires_grad)
                              for value in tensors), tuple(settings))
            if self.identity is None:
                self.identity = identity
            elif self.identity != identity:
                raise RuntimeError("Captured model/optimizer storage or hyperparameters changed; recreate the capture")
            # Actor improvement clears WM .grad attributes between updates.
            # Reattach the captured buffers; recorded backward kernels overwrite them.
            for parameter, grad in zip(self.module.parameters(), self.grads, strict=True):
                parameter.grad = grad
            self.graph.replay()
            return self.outputs.clone()
