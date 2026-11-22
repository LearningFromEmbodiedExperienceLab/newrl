"""Fetch and cook Spot and AS2 into this repository's cache.

Vendor trees come from assetx. Cooked bundles are written under
``<repo>/.cache/robots`` and are not read from another project.
"""

from pathlib import Path

from assetx import (
    ApproximateWithCapsule,
    Compose,
    GeomsToBody,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
    RenameBodies,
    ReplaceCylinderWithCapsule,
)
from assetx.cook import Recipe, cook

MENAGERIE_SPOT = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/"
    "71f066ad0be9cd271f7ed58c030243ef157af9f4/boston_dynamics_spot"
)
UNITREE_AS2 = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "38a9b00ecbff0b70e85a7627eb9c65db9a2341f7/robots/as2_description"
)

_SPOT_LEGS = ("fl", "fr", "hl", "hr")
_AS2_FEET = (
    ("FL", "FL_calf_collision3", "FL_calf_visual1"),
    ("FR", "FR_calf_collision3", "FR_calf_visual1"),
    ("RL", "RL_calf_collision3", "RL_calf_visual1"),
    ("RR", "RR_calf_collision3", "RR_calf_visual1"),
)


def build_spot(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*"]),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(),
            RenameBodies({"body": "base_link"}),
            *(
                GeomsToBody([leg.upper()], f"{leg}_foot", mass=0.05)
                for leg in _SPOT_LEGS
            ),
            ApproximateWithCapsule(
                [[f"{leg}_uleg_collision0"] for leg in _SPOT_LEGS],
                names=[f"{leg}_uleg_collision0" for leg in _SPOT_LEGS],
                replace=True,
                coverage_quantile=0.95,
                radius_scale=0.6,
            ),
            ApproximateWithCapsule(
                [[f"{leg}_lleg_collision0"] for leg in _SPOT_LEGS],
                names=[f"{leg}_lleg_collision0" for leg in _SPOT_LEGS],
                replace=True,
                coverage_quantile=0.95,
                radius_scale=0.9,
            ),
        ]
    ).transform(MujocoAsset.from_file(vendor_dir / "spot.xml"))


def build_as2(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*pos", ".*torque", "imu.*", ".*vel"]),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(),
            *(
                GeomsToBody([collision, visual], f"{prefix}_foot", mass=0.05)
                for prefix, collision, visual in _AS2_FEET
            ),
            ReplaceCylinderWithCapsule(),
        ]
    ).transform(MujocoAsset.from_file(vendor_dir / "as2.xml"))


RECIPES = {
    "spot": Recipe("spot", build_spot, MENAGERIE_SPOT),
    "as2": Recipe("as2", build_as2, UNITREE_AS2),
}


def cooked_robot(name: str, cache: Path) -> Path:
    """Cook ``name`` into ``cache`` with assetx, fetching the vendor tree if needed."""
    if name not in RECIPES:
        raise KeyError(f"No robot recipe {name!r}")
    return cook(name, cache / name, get_recipe=RECIPES.get, usd=True)


def cooked_usd(bundle: Path) -> Path:
    files = sorted((bundle / "usd").glob("*.usdc"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one USD file in {bundle / 'usd'}, found {files}")
    return files[0]
