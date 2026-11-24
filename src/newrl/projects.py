"""Manage standalone project checkouts and load installed registrations."""

import json
import keyword
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from importlib.metadata import entry_points
from pathlib import Path
from urllib.parse import urlsplit

from newrl.utils.config import cs

PROJECT_ENTRY_POINTS = "newrl.projects"
_loaded_projects: set[str] = set()


def load_projects() -> None:
    """Register presets provided by projects installed in this environment."""
    projects = sorted(
        entry_points(group=PROJECT_ENTRY_POINTS), key=lambda item: item.name
    )
    names = [project.name for project in projects]
    if len(set(names)) != len(names):
        raise ValueError("Installed NewRL projects must have distinct entry-point names")
    if projects:
        import newrl.algos  # noqa: F401
        import newrl.configs  # noqa: F401

    for project in projects:
        if project.name not in _loaded_projects:
            register = project.load()
            register(cs, project.name)
            _loaded_projects.add(project.name)


def project_directory(directory: Path | None = None) -> Path:
    """Resolve the shared parent for created and cloned projects."""
    if directory is not None:
        return directory.expanduser().resolve()
    core = Path(__file__).resolve().parents[2]
    checkout = (core / "pyproject.toml").is_file() and (core / "src/newrl").is_dir()
    return ((core.parent if checkout else Path.cwd()) / "newrl-projects").resolve()


def list_projects(directory: Path | None = None) -> list[Path]:
    """Find local project checkouts declaring a NewRL registration."""
    parent = project_directory(directory)
    if not parent.exists():
        return []
    projects = []
    for root in sorted(parent.iterdir()):
        metadata = root / "pyproject.toml"
        if not root.is_dir() or not metadata.is_file():
            continue
        try:
            config = tomllib.loads(metadata.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            continue
        if config.get("project", {}).get("entry-points", {}).get(PROJECT_ENTRY_POINTS):
            projects.append(root)
    return projects


def install_project(url: str, directory: Path | None = None) -> Path:
    """Clone a project and install it in its own Python environment."""
    parsed = urlsplit(url)
    path = parsed.path
    if url.startswith("git@") and ":" in url:
        path = url.split(":", 1)[1]
    elif parsed.scheme not in ("https", "ssh") or not parsed.hostname:
        raise ValueError("Use an HTTPS or SSH Git repository URL")
    name = path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Repository URL must end with a project name")
    parent = project_directory(directory)
    root = parent / name
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".newrl-install-", dir=parent) as temporary:
        clone = Path(temporary) / name
        subprocess.run(["git", "clone", "--", url, str(clone)], check=True)
        if clone not in list_projects(Path(temporary)):
            raise ValueError("Repository must declare a newrl.projects entry point in pyproject.toml")
        clone.rename(root)
    try:
        subprocess.run(["uv", "venv", "--python", "3.13", ".venv"], cwd=root, check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(root / ".venv/bin/python"), "-e", "."],
            cwd=root, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"Cloned project to {root}, but environment setup failed. "
            "Adjust its package sources and follow its README to finish installation."
        ) from error
    return root


def remove_project(name: str, directory: Path | None = None) -> Path:
    """Delete a named project checkout, including its environment and local data."""
    parent = project_directory(directory)
    root = parent / name
    if root.parent != parent or root.is_symlink() or root not in list_projects(parent):
        raise ValueError("Choose a project name shown by newrl project list")
    shutil.rmtree(root)
    return root


def create_project(name: str, directory: Path | None = None) -> Path:
    """Scaffold a package without installing dependencies or creating a lockfile."""
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        or keyword.iskeyword(name)
        or name.lower() == "newrl"
    ):
        raise ValueError("Use a Python package name such as foo_project")

    core = Path(__file__).resolve().parents[2]
    core_pyproject = core / "pyproject.toml"
    checkout = core_pyproject.is_file() and (core / "src/newrl").is_dir()
    root = project_directory(directory) / name

    sources = {}
    if checkout:
        sources["newrl"] = {"path": os.path.relpath(core, start=root), "editable": True}
        core_config = tomllib.loads(core_pyproject.read_text(encoding="utf-8"))
        core_sources = core_config.get("tool", {}).get("uv", {}).get("sources", {})
        for dependency, source in core_sources.items():
            if "path" in source:
                source_path = (core / source["path"]).resolve()
                if source_path.exists():
                    sources[dependency] = {
                        **source,
                        "path": os.path.relpath(source_path, start=root),
                    }
            elif "git" in source:
                sources[dependency] = source

    pyproject = f"""\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["newrl"]

[project.entry-points."{PROJECT_ENTRY_POINTS}"]
{name} = "{name}:register"

[tool.setuptools.packages.find]
where = ["src"]
"""
    if sources:
        pyproject += "\n[tool.uv.sources]\n"
        for dependency, source in sources.items():
            options = ", ".join(
                f"{key} = {json.dumps(value)}" for key, value in source.items()
            )
            pyproject += f"{dependency} = {{ {options} }}\n"

    root.mkdir(parents=True, exist_ok=False)
    package = root / "src" / name
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (root / ".gitignore").write_text(
        ".venv/\nuv.lock\n__pycache__/\n*.egg-info/\nbuild/\ndist/\n/.cache/\n/outputs_train/\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        '''"""Project presets for the NewRL commands."""

from newrl.configs import AnymalLocomotion


def register(store, project):
    store.store(
        f"{project}:AnymalLocoFlat",
        AnymalLocomotion(name=f"{project}-AnymalLocoFlat", num_envs=64),
        group="task",
    )
''',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"""\
# {name}

Create and install into this project's environment from the project directory.
Use your machine's package indexes and adjust the source paths in
`pyproject.toml` for your checkouts.

```sh
uv venv --python 3.13
uv pip install -e .
.venv/bin/newrl train task={name}:AnymalLocoFlat algo=ppo
```

Add task and algorithm presets in `src/{name}/` and register them in
`register(store, project)`. Use the project name as the preset namespace.
Add project dependencies in `pyproject.toml`. Keep this project's environment
and package sources independent of other projects.

Use the built-in train and play commands for standard NewRL tasks. If the
experiment needs a different workflow, keep the train and play scripts here.
Custom scripts can call `newrl.projects.load_projects()` before resolving presets.
""",
        encoding="utf-8",
    )
    return root
