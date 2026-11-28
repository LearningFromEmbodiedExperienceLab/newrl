"""Core robot factories and asset recipes."""

from assetx.cook import Recipe

from newrl.utils.config import cs

from .anymal import newton_anymal_c
from .g1 import MJLAB_G1, build_g1, assetx_g1
from .spot import MENAGERIE_SPOT, build_spot, assetx_spot
from .as2 import UNITREE_AS2, build_as2, assetx_as2
from .as2w import UNITREE_AS2W, build_as2w, build_as2w_sphere, assetx_as2w, assetx_as2w_sphere
from .m20s import DEEP_M20S, build_m20s, assetx_m20s
from .config import ActuatorGroup, RobotConfig, RoundedWheel, scalar_joints


for _name, _recipe in {
    "g1": Recipe("g1", build_g1, MJLAB_G1),
    "spot": Recipe("spot", build_spot, MENAGERIE_SPOT),
    "as2": Recipe("as2", build_as2, UNITREE_AS2),
    "as2w": Recipe("as2w", build_as2w, UNITREE_AS2W),
    "as2w_sphere": Recipe("as2w_sphere", build_as2w_sphere, UNITREE_AS2W),
    "m20s": Recipe("m20s", build_m20s, DEEP_M20S),
}.items():
    cs.store(_name, _recipe, group="asset")

for _name, _factory in (
    ("anymal_c", newton_anymal_c),
    ("g1", assetx_g1),
    ("spot", assetx_spot),
    ("as2", assetx_as2),
    ("as2w", assetx_as2w),
    ("as2w_sphere", assetx_as2w_sphere),
    ("m20s", assetx_m20s),
):
    cs.store(_name, _factory, group="robot")
