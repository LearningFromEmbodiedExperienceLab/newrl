"""Command-line entry point for NewRL."""

import argparse
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="newrl")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("train", "Train a policy with the built-in on-policy loop"),
        ("play", "Play a policy or inspect a task"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("overrides", nargs=argparse.REMAINDER, metavar="key=value")
    project = commands.add_parser("project", help="Manage research projects")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    create = project_commands.add_parser("create", help="Scaffold a standalone project")
    create.add_argument("name")
    create.add_argument("--dir", type=Path, help="Parent directory for the new project")
    install = project_commands.add_parser("install", help="Clone a project and install its environment")
    install.add_argument("url", help="HTTPS or SSH Git repository URL")
    install.add_argument("--dir", type=Path, help="Parent directory for the project")
    listing = project_commands.add_parser("list", help="List local project checkouts")
    listing.add_argument("--dir", type=Path, help="Parent directory containing projects")
    remove = project_commands.add_parser("remove", help="Delete a project checkout and its local data")
    remove.add_argument("name", help="Directory name shown by project list")
    remove.add_argument("--dir", type=Path, help="Parent directory containing projects")
    remove.add_argument("-y", "--yes", action="store_true", help="Confirm deletion without prompting")
    args = parser.parse_args(argv)

    if args.command in ("train", "play"):
        from newrl.utils.config import ConfigError

        if args.command == "train":
            from newrl.commands.train import main as run
        else:
            from newrl.commands.play import main as run
        try:
            run(args.overrides)
        except ConfigError as error:
            parser.error(str(error))
    else:
        from newrl.projects import create_project, install_project, list_projects, remove_project

        try:
            if args.project_command == "create":
                root = create_project(args.name, args.dir)
                print(f"Created project {root}")
                print("See its README for environment setup and training.")
            elif args.project_command == "install":
                root = install_project(args.url, args.dir)
                print(f"Installed project {root}")
                print(f"Use {root / '.venv/bin/newrl'} for its train and play commands.")
            elif args.project_command == "list":
                roots = list_projects(args.dir)
                for root in roots:
                    print(f"{root.name}\t{root}")
                if not roots:
                    print("No local projects found.")
            else:
                roots = {root.name: root for root in list_projects(args.dir)}
                if args.name not in roots:
                    raise ValueError("Choose a project name shown by newrl project list")
                if not args.yes:
                    answer = input(
                        f"Delete {roots[args.name]}, including its environment, assets, "
                        "and uncommitted work? [y/N] "
                    )
                    if answer.strip().lower() not in ("y", "yes"):
                        print("Removal cancelled.")
                        return
                print(f"Removed project {remove_project(args.name, args.dir)}")
        except (OSError, ValueError, subprocess.CalledProcessError, EOFError) as error:
            project.error(str(error))


if __name__ == "__main__":
    main()
