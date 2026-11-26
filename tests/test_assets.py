"""Ready bundles stay usable offline; rebuilds are explicit."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import ANY, call, patch

from newrl.assets import cooking
from newrl.cli import main
from newrl.utils.config import cs


class AssetCacheTests(unittest.TestCase):
    def bundle(self, root, name='as2w'):
        bundle = root / name
        (bundle / 'usd').mkdir(parents=True)
        (bundle / 'model.xml').write_text('<mujoco/>')
        (bundle / 'model.urdf').write_text('<robot name="test"/>')
        (bundle / 'usd' / 'robot.usdc').write_bytes(b'cached USD')
        (bundle / 'assetx.json').write_text(json.dumps({
            'recipe':name, 'hash':'old source hash', 'formats':['mjcf','urdf','usd'],
        }))
        return bundle

    def test_ready_bundle_survives_source_changes_without_cooking(self):
        with TemporaryDirectory() as directory, patch.object(cooking, 'cook', side_effect=AssertionError('unexpected rebuild')):
            root = Path(directory)
            for name in cs.names("asset"):
                with self.subTest(name=name):
                    bundle = self.bundle(root,name)
                    self.assertEqual(cooking.cooked_robot(name,root),bundle)

    def test_missing_incomplete_and_invalid_bundles_are_repaired(self):
        for missing in ('assetx.json','model.xml','model.urdf','usd/robot.usdc'):
            with self.subTest(missing=missing), TemporaryDirectory() as directory, patch.object(cooking,'cook') as cook:
                root = Path(directory)
                bundle = self.bundle(root)
                (bundle / missing).unlink()
                cooking.cooked_robot('as2w',root)
                cook.assert_called_once_with('as2w',bundle,get_recipe=ANY,usd=True,force=True)
                self.assertEqual(cook.call_args.kwargs['get_recipe']('as2w'), cs.get('asset', 'as2w'))
        for manifest in ('invalid JSON','[]','{"recipe":"spot","formats":["usd"]}',
                         '{"recipe":"as2w","formats":null}'):
            with self.subTest(manifest=manifest), TemporaryDirectory() as directory, patch.object(cooking,'cook') as cook:
                root = Path(directory)
                bundle = self.bundle(root)
                (bundle / 'assetx.json').write_text(manifest)
                cooking.cooked_robot('as2w',root)
                cook.assert_called_once()
        with TemporaryDirectory() as directory, patch.object(cooking,'cook') as cook:
            cooking.cooked_robot('spot',Path(directory))
            cook.assert_called_once()

    def test_explicit_rebuild_bypasses_ready_bundle(self):
        with TemporaryDirectory() as directory, patch.object(cooking,'cook') as cook:
            root = Path(directory)
            bundle = self.bundle(root)
            cooking.cooked_robot('as2w',root,rebuild=True)
            cook.assert_called_once_with('as2w',bundle,get_recipe=ANY,usd=True,force=True)

    def test_rebuild_command_validates_names_before_modifying_cache(self):
        with patch.object(cooking,'cooked_robot') as cook, patch('newrl.assets.cooking.robot_model_dir',return_value=Path('/tmp/robots')), redirect_stdout(StringIO()):
            main(['assets','rebuild','spot','as2w','spot'])
            self.assertEqual(cook.call_args_list,[call('spot',Path('/tmp/robots'),rebuild=True),call('as2w',Path('/tmp/robots'),rebuild=True)])
        with patch.object(cooking,'cooked_robot') as cook, redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(['assets','rebuild','spot','unknown'])
        cook.assert_not_called()


if __name__ == '__main__':
    unittest.main()
