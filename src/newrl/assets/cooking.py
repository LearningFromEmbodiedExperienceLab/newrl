"""Resolve and cook cached robot bundles."""

import json
import os
from pathlib import Path

from assetx.cook import Recipe, cook

from newrl.utils.config import cs


def robot_model_dir() -> Path:
    """Cooked robot bundles for this repository.

    ``ROBOT_MODEL_DIR`` overrides the default ``<repo>/.cache/robots``.
    """
    if env := os.environ.get("ROBOT_MODEL_DIR"):
        return Path(env)
    return Path(__file__).resolve().parents[3] / ".cache" / "robots"


def cooked_robot(name: str, cache: Path, *, rebuild: bool = False) -> Path:
    """Reuse complete bundles; fetch/cook only missing bundles or explicit rebuilds.

    Assetx hashes the entire recipe module and its library sources. That hash
    must not invalidate ready robot assets during ordinary task startup.
    """
    recipes = cs.definitions("asset")
    if name not in recipes:
        raise KeyError(f"No robot recipe {name!r}. Known: {sorted(recipes)}")
    if not isinstance(recipes[name], Recipe):
        raise TypeError(f"Asset preset {name!r} must be an assetx Recipe")
    if any(not part or part in (".", "..") or "/" in part or "\\" in part for part in name.split(":")):
        raise ValueError(f"Invalid asset name {name!r}")
    bundle = Path(cache).joinpath(*name.split(":"))
    if not rebuild:
        try:
            manifest = json.loads((bundle / 'assetx.json').read_text())
            complete = (
                isinstance(manifest, dict)
                and manifest.get('recipe') == name
                and isinstance(manifest.get('formats'), list)
                and 'usd' in manifest.get('formats', [])
                and (bundle / 'model.xml').is_file()
                and (bundle / 'model.urdf').is_file()
                and len([path for path in (bundle / 'usd').glob('*.usdc') if path.is_file()]) == 1
            )
            if complete:
                return bundle
        except (OSError, ValueError):
            pass
    # Force also repairs incomplete bundles whose source hash is still current.
    return cook(name, bundle, get_recipe=recipes.__getitem__, usd=True, force=True)


def cooked_usd(bundle: Path) -> Path:
    files = sorted((bundle / "usd").glob("*.usdc"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one USD file in {bundle / 'usd'}, found {files}")
    return files[0]
