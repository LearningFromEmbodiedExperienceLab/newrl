"""Command-line entry point for NewRL."""

import argparse
import subprocess
from importlib import import_module
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newrl")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description, module, schema in (
        ("train", "Train a policy with the built-in on-policy loop", "train", "TrainConfig"),
        ("play", "Play a policy or inspect a task", "play", "PlayConfig"),
        ("play-timeline", "Play, rewind, and branch a single world", "play_timeline", "TimelinePlayConfig"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("overrides", nargs=argparse.REMAINDER, metavar="key=value")
        command.set_defaults(config_command=f"newrl.commands.{module}", config_schema=schema)
    assets = commands.add_parser('assets', help='Manage cached robot assets')
    asset_commands = assets.add_subparsers(dest='assets_command', required=True)
    asset_commands.add_parser("list", help="List registered cooking recipes")
    robots = commands.add_parser("robots", help="Discover registered robot factories")
    robots.add_subparsers(dest="robots_command", required=True).add_parser("list")
    rebuild = asset_commands.add_parser('rebuild', help='Explicitly rebuild cooked robot bundles')
    rebuild.add_argument('names', nargs='+', metavar='ROBOT')
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "config_command"):
        from newrl.utils.config import ConfigError

        try:
            import_module(args.config_command).main(args.overrides)
        except ConfigError as error:
            parser.error(str(error))
    elif args.command in ("assets", "robots"):
        from newrl.projects import load_projects
        from newrl.assets import cooked_robot
        from newrl.robots import robot_model_dir
        from newrl.utils.config import cs

        load_projects()
        group = "asset" if args.command == "assets" else "robot"
        names = cs.names(group)
        if args.command == "robots" or args.assets_command == "list":
            for name in names:
                print(name)
        else:
            unknown = set(args.names) - set(names)
            if unknown:
                parser.error(f"Unknown robots: {', '.join(sorted(unknown))}. Choose from: {', '.join(names)}")
            for name in dict.fromkeys(args.names):
                print(f"Rebuilt {name}: {cooked_robot(name, robot_model_dir(), rebuild=True)}")
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
            parser.error(str(error))


if __name__ == "__main__":
    main()
