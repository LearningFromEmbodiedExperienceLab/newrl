"""Allowlisted metadata for shareable benchmark results."""

import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

import newton
import torch


def _query(*args):
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _revision(directory):
    commit = _query("git", "-C", str(directory), "rev-parse", "HEAD")
    if commit is None or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return None
    status = _query("git", "-C", str(directory), "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _cpu_model():
    try:
        with Path("/proc/cpuinfo").open() as cpuinfo:
            for line in cpuinfo:
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    return value.strip()
    except OSError:
        pass
    return None


def collect_metadata(device, machine_id):
    """Collect hardware classes and versions without machine or user identifiers."""
    gpu = torch.cuda.get_device_properties(device)
    driver = _query("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    driver_versions = sorted(set(driver.splitlines())) if driver else []
    versions = {}
    for package in (
        "newton",
        "warp-lang",
        "mujoco",
        "mujoco-warp",
        "torch",
        "tensordict",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "machine": {
            "id": machine_id,
            "os": platform.system(),
            "architecture": platform.machine(),
            "cpu_model": _cpu_model(),
            "gpu_model": gpu.name,
            "gpu_memory_bytes": gpu.total_memory,
            "gpu_compute_capability": [gpu.major, gpu.minor],
        },
        "software": {
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
            "cuda_driver_versions": driver_versions,
            "packages": versions,
        },
        "revisions": {
            "newrl": _revision(Path(__file__).resolve().parents[1]),
            "newton": _revision(Path(newton.__file__).resolve().parent),
        },
    }
