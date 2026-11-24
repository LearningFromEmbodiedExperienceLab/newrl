"""Lazily fetch and cook robot assets into this repository's cache.

Vendor trees come from assetx. Cooked bundles are written under
``<repo>/.cache/robots`` and are not read from another project.
"""

from pathlib import Path
from dataclasses import replace
import tempfile
import mujoco

from assetx import (
    ApproximateWithCapsule,
    AddJoint,
    Compose,
    EditBodies,
    GeomsToBody,
    JointCfg,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveGeoms,
    RemoveSceneSettings,
    RemoveSensors,
    RenameBodies,
    ReplaceCylinderWithCapsule,
    Transform,
)
from assetx.cook import Recipe, cook
from assetx.conversion.urdf2mjcf import urdf_to_mjcf

MENAGERIE_SPOT = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/"
    "71f066ad0be9cd271f7ed58c030243ef157af9f4/boston_dynamics_spot"
)
UNITREE_AS2 = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "38a9b00ecbff0b70e85a7627eb9c65db9a2341f7/robots/as2_description"
)
UNITREE_AS2W = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48/robots/as2w_description"
)
DEEP_M20S = (
    "https://github.com/DeepRoboticsLab/deep_robotics_model/tree/"
    "a547a0289deb7cda23578b252b4d54debfdcec19/M20S/mjcf"
)
MJLAB_G1 = (
    "https://github.com/mujocolab/mjlab/tree/"
    "1f48001802bd89257537c2dc16979d41b9686e86/"
    "src/mjlab/asset_zoo/robots/unitree_g1/xmls"
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


class _RemoveUnnamedSensors(Transform):
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for sensor in list(spec.sensors):
            if not sensor.name:
                spec.delete(sensor)
        spec.compile()
        return replace(asset, spec=spec)


class _M20SCapsules(Transform):
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and "_wheel" not in geom.name:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        return replace(asset, spec=spec)


def build_m20s(vendor_dir: Path) -> MujocoAsset:
    return Compose([
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        _RemoveUnnamedSensors(),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        RemoveGeoms(["floor"]),
        _M20SCapsules(),
    ]).transform(MujocoAsset.from_file(vendor_dir / "M20S.xml"))


def build_as2w(vendor_dir: Path, *, sphere_wheels: bool = False) -> MujocoAsset:
    xml = Path(tempfile.mkdtemp(prefix="newrl-as2w-")) / "as2w.xml"
    urdf_to_mjcf(vendor_dir / "as2w.urdf", out_xml=xml)
    transforms = [
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        AddJoint("base_link", JointCfg(name="floating_base_joint", type="free")),
        ApproximateWithCapsule(
            [[f"{leg}_calf_collision0"] for leg in ("FL", "FR", "RL", "RR")],
            names=[f"{leg}_calf_collision0" for leg in ("FL", "FR", "RL", "RR")],
            replace=True,
            coverage_quantile=0.95,
            radius_scale=0.9,
        ),
    ]
    if sphere_wheels:
        from assetx import ApproximateWithSphere

        wheels = [f"{leg}_foot_collision0" for leg in ("FL", "FR", "RL", "RR")]
        transforms.append(
            ApproximateWithSphere([[wheel] for wheel in wheels], names=wheels, replace=True)
        )
    return Compose(transforms).transform(MujocoAsset.from_file(xml))


def build_as2w_sphere(vendor_dir: Path) -> MujocoAsset:
    return build_as2w(vendor_dir, sphere_wheels=True)


def build_g1(vendor_dir: Path) -> MujocoAsset:
    return Compose([
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        # RobotConfig.position owns the root placement.
        EditBodies("pelvis", pos=[0.0, 0.0, 0.0]),
    ]).transform(MujocoAsset.from_file(vendor_dir / "g1.xml"))


RECIPES = {
    "g1": Recipe("g1", build_g1, MJLAB_G1),
    "spot": Recipe("spot", build_spot, MENAGERIE_SPOT),
    "as2": Recipe("as2", build_as2, UNITREE_AS2),
    "as2w": Recipe("as2w", build_as2w, UNITREE_AS2W),
    "as2w_sphere": Recipe("as2w_sphere", build_as2w_sphere, UNITREE_AS2W),
    "m20s": Recipe("m20s", build_m20s, DEEP_M20S),
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
