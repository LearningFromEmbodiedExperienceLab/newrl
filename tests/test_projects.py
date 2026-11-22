"""Project scaffolding and entry-point discovery."""

import contextlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from newrl import projects
from newrl.cli import main
from newrl.utils.config import ConfigStore


class ProjectTests(unittest.TestCase):
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
