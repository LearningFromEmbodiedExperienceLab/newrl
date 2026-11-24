"""Project scaffolding and entry-point discovery."""

import contextlib
import io
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from newrl import projects
from newrl.cli import main
from newrl.utils.config import ConfigStore


class ProjectTests(unittest.TestCase):
    def test_install_https_and_ssh_into_isolated_environment(self):
        for url in ("https://github.com/team/foo-project.git", "git@github.com:team/foo-project.git",
                    "ssh://git@github.com/team/foo-project.git"):
            with self.subTest(url=url), tempfile.TemporaryDirectory() as directory:
                def run(command, **kwargs):
                    if command[:2] == ["git", "clone"]:
                        clone = Path(command[-1])
                        clone.mkdir()
                        (clone / "pyproject.toml").write_text(
                            '[project.entry-points."newrl.projects"]\nfoo = "foo:register"\n'
                        )
                with patch.object(projects.subprocess, "run", side_effect=run) as execute:
                    root = projects.install_project(url, Path(directory))
                self.assertEqual(root, Path(directory) / "foo-project")
                self.assertEqual(projects.list_projects(Path(directory)), [root])
                self.assertEqual(execute.call_args_list[1].kwargs["cwd"], root)
                self.assertEqual(execute.call_args_list[2].args[0], [
                    "uv", "pip", "install", "--python", str(root / ".venv/bin/python"), "-e", ".",
                ])
                with self.assertRaises(FileExistsError):
                    projects.install_project(url, Path(directory))

    def test_invalid_clone_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(command, **kwargs):
                Path(command[-1]).mkdir()
            with patch.object(projects.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "entry point"):
                    projects.install_project("https://github.com/team/foo.git", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_setup_retains_checkout_for_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(command, **kwargs):
                if command[0] == "git":
                    clone = Path(command[-1])
                    clone.mkdir()
                    (clone / "pyproject.toml").write_text(
                        '[project.entry-points."newrl.projects"]\nfoo = "foo:register"\n'
                    )
                else:
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(projects.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "environment setup failed"):
                    projects.install_project("https://github.com/team/foo.git", Path(directory))
            self.assertEqual(projects.list_projects(Path(directory)), [Path(directory) / "foo"])

    def test_remove_confirmation_and_path_safety(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            parent = Path(directory)
            root = projects.create_project("foo", parent)
            with patch("builtins.input", return_value="n"):
                main(["project", "remove", "foo", "--dir", directory])
            self.assertTrue(root.is_dir())
            link = parent / "linked"
            link.symlink_to(root, target_is_directory=True)
            for name in ("../foo", "linked", "."):
                with self.assertRaises(ValueError):
                    projects.remove_project(name, parent)
            self.assertTrue(root.is_dir())
            main(["project", "remove", "foo", "--dir", directory, "--yes"])
            self.assertFalse(root.exists())

    def test_create_writes_package_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ):
            main(["project", "create", "foo_project", "--dir", directory])
            root = Path(directory) / "foo_project"
            metadata = tomllib.loads((root / "pyproject.toml").read_text())
            self.assertEqual(metadata["project"]["dependencies"], ["newrl"])
            self.assertEqual(
                metadata["project"]["entry-points"]["newrl.projects"]["foo_project"],
                "foo_project:register",
            )
            self.assertTrue((root / "src/foo_project/__init__.py").is_file())
            self.assertFalse((root / "uv.lock").exists())
            self.assertIn("uv.lock", (root / ".gitignore").read_text().splitlines())
            with self.assertRaises(FileExistsError):
                projects.create_project("foo_project", Path(directory))

    def test_load_registers_installed_project_once(self):
        entry_point = Mock()
        entry_point.name = "foo_project"
        store = ConfigStore()
        with (
            patch.object(projects, "cs", store),
            patch.object(projects, "_loaded_projects", set()),
            patch.object(projects, "entry_points", return_value=[entry_point]) as discover,
        ):
            projects.load_projects()
            projects.load_projects()
        discover.assert_called_with(group="newrl.projects")
        entry_point.load.return_value.assert_called_once_with(store, "foo_project")
