from pathlib import Path

import newton
import yaml

from .config import RobotConfig


def newton_anymal_c() -> RobotConfig:
    asset = Path(newton.utils.download_asset("anybotics_anymal_c"))
    with (asset / "rl_policies" / "anymal.yaml").open() as file:
        cfg = yaml.safe_load(file)
    return RobotConfig(
        asset=asset / "usd" / "anymal_c.usda",
        position=(0.0, 0.0, 0.76),
        joint_position=tuple(cfg["mjw_joint_pos"]),
        stiffness=tuple(cfg["mjw_joint_stiffness"]),
        damping=tuple(cfg["mjw_joint_damping"]),
        armature=tuple(cfg["mjw_joint_armature"]),
        action_scale=float(cfg["action_scale"]),
        joint_mirror={
            "LF_HAA": ("RF_HAA", -1),
            "RF_HAA": ("LF_HAA", -1),
            "LF_HFE": ("RF_HFE", 1),
            "RF_HFE": ("LF_HFE", 1),
            "LF_KFE": ("RF_KFE", 1),
            "RF_KFE": ("LF_KFE", 1),
            "LH_HAA": ("RH_HAA", -1),
            "RH_HAA": ("LH_HAA", -1),
            "LH_HFE": ("RH_HFE", 1),
            "RH_HFE": ("LH_HFE", 1),
            "LH_KFE": ("RH_KFE", 1),
            "RH_KFE": ("LH_KFE", 1),
        },
        body_mirror={
            "base": ("base", 1),
            "LF_FOOT": ("RF_FOOT", 1),
            "RF_FOOT": ("LF_FOOT", 1),
            "LH_FOOT": ("RH_FOOT", 1),
            "RH_FOOT": ("LH_FOOT", 1),
        },
    )
