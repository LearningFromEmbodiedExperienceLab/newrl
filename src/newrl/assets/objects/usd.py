from dataclasses import dataclass

import newton
import warp as wp


@dataclass(frozen=True)
class USDObjectConfig:
    asset: str
    position: tuple[float, float, float] = (0, 0, 0)
    orientation: tuple[float, float, float, float] = (0, 0, 0, 1)
    fixed: bool | None = None

    def build(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(self.asset, xform=wp.transform(self.position, self.orientation),
                        floating=None if self.fixed is None else not self.fixed,
                        joint_drive_gains_scaling=0.0)
        for dof in range(builder.joint_dof_count):
            builder.joint_target_ke[dof] = 0.0
            builder.joint_target_kd[dof] = 0.0
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
        return builder
