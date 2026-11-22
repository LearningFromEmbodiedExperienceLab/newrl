"""Resolve local and W&B run checkpoints for playback."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

_RUN_PART = re.compile(r"[A-Za-z0-9_.-]+")
_CHECKPOINT = re.compile(r"checkpoint_([0-9]+)\.pt")


def resolve_checkpoint(spec: str, cache_dir: Path) -> Path:
    """Fetch the latest uploaded checkpoint or a pinned run iteration."""
    if not spec.startswith("run:"):
        return Path(spec).expanduser()

    run_path, separator, iteration = spec.removeprefix("run:").partition(":")
    parts = run_path.split("/")
    if len(parts) != 3 or any(
        part in (".", "..") or not _RUN_PART.fullmatch(part) for part in parts
    ):
        raise ValueError("Expected run:<entity>/<project>/<run_id>[:<iteration>]")
    if separator and (not iteration.isascii() or not iteration.isdecimal()):
        raise ValueError("W&B checkpoint iteration must be a non-negative integer")
    selected_iteration = int(iteration) if separator else None

    import wandb

    run = wandb.Api().run(run_path)
    selected = None
    latest_iteration = -1
    for file in run.files(pattern="checkpoint_%.pt"):
        match = _CHECKPOINT.fullmatch(file.name)
        if match is None or file.size <= 0:
            continue
        step = int(match[1])
        if selected_iteration is not None and step != selected_iteration:
            continue
        if step > latest_iteration:
            selected, latest_iteration = file, step
    if selected is None:
        label = (
            "checkpoint" if selected_iteration is None else f"checkpoint {selected_iteration}"
        )
        raise FileNotFoundError(f"No uploaded {label} for W&B run {run_path}")

    directory = cache_dir.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / selected.name
    if path.is_file() and path.stat().st_size == selected.size:
        return path
    with TemporaryDirectory(dir=directory) as temporary:
        with selected.download(root=temporary):
            downloaded = Path(temporary) / selected.name
        if downloaded.stat().st_size != selected.size:
            raise OSError(
                f"Incomplete download of {selected.name} from W&B run {run_path}"
            )
        downloaded.replace(path)
    return path
