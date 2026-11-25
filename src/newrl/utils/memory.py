"""Sample CUDA memory at training phase boundaries."""

from typing import TYPE_CHECKING

from newrl.algos.interfaces import Learner

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
import warp as wp
from tensordict import TensorDictBase, TensorDict

if TYPE_CHECKING:
    from newrl.env import _EnvironmentCore

_MIB = 1024**2


def storage_bytes(*values: object, seen: set[int] | None  =  None) -> int:
    """Count identifiable CUDA allocations, including Warp-owned arrays."""
    if seen is None:
        seen = set()
    total = 0
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            pointer, size = storage.data_ptr(), storage.nbytes()
        elif isinstance(value, wp.array) and value.device.is_cuda and value.deleter:
            pointer, size = value.ptr, value.capacity
        else:
            if isinstance(value, wp.Mesh):
                total += storage_bytes(
                    value.points, value.indices, value.velocities, seen=seen
                )
            elif isinstance(value, Mapping | TensorDictBase):
                total += storage_bytes(*value.values(), seen=seen)
            elif isinstance(value, tuple | list):
                total += storage_bytes(*value, seen=seen)
            elif is_dataclass(value) and not isinstance(value, type):
                total += storage_bytes(
                    *(getattr(value, item.name) for item in fields(value)), seen=seen
                )
            continue
        if pointer and pointer not in seen:
            seen.add(pointer)
            total += size
    return total


class CudaMemoryProfiler:
    def __init__(self, device: torch.device) -> None:
        torch.cuda.init()
        self.device = device

    def begin_phase(self) -> None:
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def end_phase(self, name: str) -> dict[str, float]:
        stats = torch.cuda.memory_stats(self.device)
        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        peak = stats.get("allocated_bytes.all.peak", 0)
        free, total = torch.cuda.mem_get_info(self.device)
        prefix = f"memory/{name}"
        return {
            f"{prefix}/torch_start_mib": self.start_allocated / _MIB,
            f"{prefix}/torch_live_mib": allocated / _MIB,
            f"{prefix}/torch_peak_mib": peak / _MIB,
            f"{prefix}/torch_peak_extra_mib": (peak - self.start_allocated) / _MIB,
            f"{prefix}/torch_reserved_mib": reserved / _MIB,
            f"{prefix}/torch_peak_reserved_mib": (
                stats.get("reserved_bytes.all.peak", 0) / _MIB
            ),
            f"{prefix}/device_used_mib": (total - free) / _MIB,
            f"{prefix}/device_outside_torch_reserved_mib": (
                total - free - reserved
            ) / _MIB,
            f"{prefix}/torch_cache_mib": (reserved - allocated) / _MIB,
        }

    def environment(self, env: "_EnvironmentCore") -> dict[str, float]:
        """Inventory exposed buffers. Native workspaces can live outside this inventory."""
        seen: set[int] = set()
        model = storage_bytes(vars(env.model), seen=seen)
        state = storage_bytes(
            vars(env.state), vars(env.state_next), vars(env.control),
            env.mdp_state, env.scene.entities, env.scene.spawn_locations,
            *(vars(sensor) for sensor in env.scene.sensors.values()),
            seen=seen,
        )
        solver_objects = (
            env.contacts, env.collision, env._solver, env.sim,
            getattr(env.collision, "broad_phase", None),
            getattr(env.collision, "narrow_phase", None),
            getattr(env.sim, "mjw_model", None), getattr(env.sim, "mjw_data", None),
            env.observables, env._solver.contact_observables,
        )
        solver = storage_bytes(
            *(vars(obj) for obj in solver_objects if hasattr(obj, "__dict__")),
            seen=seen,
        )
        return {
            "memory/buffers/model_arrays_mib": model / _MIB,
            "memory/buffers/env_state_mib": state / _MIB,
            "memory/buffers/solver_collision_arrays_mib": solver / _MIB,
        }

    def learner(self, policy: Learner) -> dict[str, float]:
        seen: set[int] = set()
        metrics = {}
        if isinstance(policy, torch.nn.Module):
            weights = storage_bytes(
                *policy.parameters(), *policy.buffers(), seen=seen
            )
            gradients = storage_bytes(
                *(param.grad for param in policy.parameters()), seen=seen
            )
            metrics["memory/buffers/policy_mib"] = weights / _MIB
            metrics["memory/buffers/gradients_mib"] = gradients / _MIB
        optimizer = getattr(policy, "optimizer", None)
        if optimizer is not None:
            metrics["memory/buffers/optimizer_mib"] = (
                storage_bytes(optimizer.state_dict(), seen=seen) / _MIB
            )
        replay = getattr(policy, "replay", None)
        if replay is not None:
            metrics["memory/buffers/replay_mib"] = storage_bytes(replay.storage, seen=seen) / _MIB
        optimizers = getattr(policy, "optimizers", ())
        if optimizers:
            metrics["memory/buffers/optimizer_mib"] = storage_bytes(
                *(opt.state_dict() for opt in optimizers), seen=seen
            ) / _MIB
        return metrics

    def rollout(self, batch: TensorDict, carry: TensorDict) -> dict[str, float]:
        seen: set[int] = set()
        return {
            "memory/buffers/rollout_mib": storage_bytes(batch, seen=seen) / _MIB,
            "memory/buffers/carry_mib": storage_bytes(carry, seen=seen) / _MIB,
        }


def write_memory_chart(path: Path, samples: list[dict[str, float]]) -> None:
    """Write a standalone SVG from the scalar samples already collected."""
    steady = [sample for sample in samples if not sample["memory/warmup"]]
    snapshot = (steady or samples)[-1]
    buffer_labels = {
        "model_arrays": "Scene and terrain",
        "env_state": "Environment state",
        "solver_collision_arrays": "Solver and collision",
        "rollout": "Rollout batch",
        "carry": "Rollout carry",
        "policy": "Policy and normalizers",
        "gradients": "Gradients",
        "optimizer": "Optimizer state",
    }
    buffers = [
        (label, snapshot[key])
        for name, label in buffer_labels.items()
        if (key := f"memory/buffers/{name}_mib") in snapshot
    ]
    peaks = []
    for label, group in (
        ("Warmup", [sample for sample in samples if sample["memory/warmup"]]),
        ("After warmup", steady),
    ):
        if group:
            for phase in ("rollout", "update"):
                peak = max(sample[f"memory/{phase}/torch_peak_mib"] for sample in group)
                peaks.append((f"{label} {phase}", peak))

    elements = [
        '<text x="24" y="40" font-size="24" font-weight="bold">GPU memory</text>',
    ]
    y = 80
    snapshot_label = "after warmup" if steady else "during warmup"
    for title, rows, color in (
        (f"Identified buffers {snapshot_label}", buffers, "#2563eb"),
        ("Peak PyTorch allocations by phase", peaks, "#0f766e"),
    ):
        elements.append(f'<text x="24" y="{y}" font-weight="bold">{title}</text>')
        scale = max((value for _, value in rows), default=0) or 1
        for label, value in rows:
            y += 28
            width = 350 * value / scale
            elements.extend((
                f'<text x="24" y="{y}">{label}</text>',
                f'<rect x="245" y="{y - 12}" width="{width:.2f}" height="14" '
                f'rx="2" fill="{color}"/>',
                f'<text x="615" y="{y}">{value:,.2f} MiB</text>',
            ))
        y += 44
    elements.append(
        f'<text x="24" y="{y}" font-size="13" fill="#475569">'
        "Buffer inventory is partial. Phase peaks include live buffers and temporary allocations."
        "</text>"
    )
    height = y + 24
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" '
        f'viewBox="0 0 760 {height}" font-family="sans-serif" font-size="14" fill="#0f172a">\n'
        f'<rect width="760" height="{height}" fill="white"/>\n'
        + "\n".join(elements)
        + "\n</svg>\n",
        encoding="utf-8",
    )
