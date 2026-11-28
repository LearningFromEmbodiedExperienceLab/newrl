"""Resolve local and W&B run checkpoints for playback."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread

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
    from tqdm import tqdm

    print(f"Resolving W&B checkpoint for {run_path}...", flush=True)
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
    print(
        f"Downloading {selected.name} ({selected.size / 1_000_000:.1f} MB)...",
        flush=True,
    )
    with TemporaryDirectory(dir=directory) as temporary:
        downloaded = _download_with_progress(selected, Path(temporary), tqdm)
        if downloaded.stat().st_size != selected.size:
            raise OSError(
                f"Incomplete download of {selected.name} from W&B run {run_path}"
            )
        downloaded.replace(path)
    return path


def _download_with_progress(file, root: Path, tqdm) -> Path:
    """Download ``file`` into ``root`` while showing bytes written."""
    destination = root / file.name
    stop = Event()

    def watch() -> None:
        seen = 0
        with tqdm(
            total=file.size,
            unit="B",
            unit_scale=True,
            unit_divisor=1000,
            desc=file.name,
            leave=False,
        ) as bar:
            while not stop.wait(0.2):
                seen = _advance(bar, destination, seen)
            _advance(bar, destination, seen)

    watcher = Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with file.download(root=root):
            pass
    finally:
        stop.set()
        watcher.join()
    return destination


def _advance(bar, path: Path, seen: int) -> int:
    current = path.stat().st_size if path.is_file() else 0
    if current > seen:
        bar.update(current - seen)
    return max(current, seen)
