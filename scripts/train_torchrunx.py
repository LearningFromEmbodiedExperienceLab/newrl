"""Experimental single-node torchrunx launcher for NewRL training."""

import ast
import logging
import os
import sys


def main(argv: list[str]) -> None:
    if "-h" in argv or "--help" in argv:
        print("Experimental: python scripts/train_torchrunx.py gpu_ids=all|[0,1] task=... algo=...")
        return
    selectors = [arg.split("=", 1)[1] for arg in argv if arg.startswith("gpu_ids=")]
    if len(selectors) > 1:
        raise ValueError("Specify gpu_ids once")
    selection = selectors[0] if selectors else "all"
    overrides = [arg for arg in argv if not arg.startswith("gpu_ids=")]

    import torch
    import torchrunx

    from newrl.commands.train import main as train

    count = torch.cuda.device_count()
    indices = list(range(count)) if selection == "all" else ast.literal_eval(selection)
    if type(indices) is int:
        indices = [indices]
    if (
        not isinstance(indices, (list, tuple))
        or not indices
        or any(type(index) is not int or not 0 <= index < count for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(f"gpu_ids must select distinct visible GPUs (0 to {count - 1})")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else list(map(str, range(count)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[index] for index in indices)
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("TORCHRUNX_LOG_DIR", "outputs_train/torchrunx")
    # NewRL initializes NCCL in each worker.
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=len(indices),
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + (
            "CUDA_VISIBLE_DEVICES", "NEWRL*", "ASSETX*", "WARP*", "WANDB*", "MUJOCO*",
            "TORCHRUNX_LOG_DIR",
        ),
    ).run(train, [*overrides, "task.device=cuda:0"])


if __name__ == "__main__":
    main(sys.argv[1:])
