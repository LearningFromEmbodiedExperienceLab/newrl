"""Command-line entry point for NewRL."""

import argparse
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
    args = parser.parse_args(argv)

    if args.command == "train":
        from newrl.commands.train import main as train

        train(args.overrides)
    elif args.command == "play":
        from newrl.commands.play import main as play

        play(args.overrides)
    else:
        from newrl.projects import create_project

        try:
            root = create_project(args.name, args.dir)
        except (OSError, ValueError) as error:
            create.error(str(error))
        print(f"Created project {root}")
        print("See its README for environment setup and training.")


if __name__ == "__main__":
    main()
