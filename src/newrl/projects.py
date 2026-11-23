"""Create standalone extension projects and load installed registrations."""

import json
import keyword
import os
import re
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

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
    if directory is None:
        directory = (core.parent if checkout else Path.cwd()) / "newrl-projects"
    root = directory.expanduser().resolve() / name

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
