# Research projects

## Create and manage projects

Install NewRL in your current environment to expose the `newrl` command.
From an existing NewRL development environment, refresh the entry point
with `uv pip install --no-deps -e .`.

```sh
newrl project create foo_project
newrl project install https://github.com/team/foo_project.git
newrl project install git@github.com:team/foo_project.git
newrl project list
newrl project remove foo_project
```

With an editable NewRL checkout, this creates the project under the
`newrl-projects/` directory beside the checkout. With an installed package,
the default is `newrl-projects/` under the current directory. Use `--dir`
to choose the parent directory.

`project install` clones the repository, creates its own Python 3.13 `.venv`,
and installs the project with `uv pip install -e .`. The repository must declare
a `newrl.projects` entry point. Use its `.venv/bin/newrl` for train and play.
If environment setup fails, the checkout is retained so its dependency sources
can be adjusted before retrying setup.
`project list` shows project checkouts under the selected parent.
`project remove` prompts before deleting a checkout, including its environment,
assets, and uncommitted work. Pass `--yes` to confirm in scripts.
All project commands accept `--dir` to select their parent directory.

Follow the generated README to create the project's own environment and
install the package. Each project chooses its dependency sources and indexes
and can use `uv pip install` without a lockfile. NewRL's train and play
commands discover presets from projects installed in the active environment
through the `newrl.projects` entry-point group.

```sh
.venv/bin/newrl train task=foo_project:AnymalLocoFlat algo=ppo
```

Run the commands from the project directory to keep outputs and downloads
there. Projects can provide their own training and playback scripts when the
built-in commands don't fit their workflow.


## Project robots and assets

See the [robot contribution guide](robots.md) for asset preparation, ownership,
factory configuration and validation.

Register an importable, zero-argument robot factory in the project callback.

```python
def register(store, project):
    store.store(f"{project}:MyRobot", my_robot, group="robot")
```

The factory returns a `RobotConfig` with a resolved USD asset path. It can use
files shipped with the project, download from a project-managed store, or
use assetx. Registration and listing are lazy. Checkpoints preserve the
factory's import path, so define it at module scope.

```sh
newrl robots list
newrl play task=AS2WSphereLocoFlat task.robots.robot=my_project:MyRobot
```

Projects using assetx can additionally register a namespaced `Recipe` under
`asset`. Set its name to the registered name and call
`cooked_robot(name, robot_model_dir())` in the factory. Cooked bundles use
namespace subdirectories. List and explicitly rebuild recipes with
`newrl assets list` and `newrl assets rebuild my_project:MyRobot`.
Projects control acquisition of all other assets through their own factories.
