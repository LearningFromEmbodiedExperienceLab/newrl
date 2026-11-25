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
