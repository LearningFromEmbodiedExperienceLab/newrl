"""Display the current motion reference using the tracked entity's meshes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .markers import RobotGhost

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torch import Tensor
    from newrl.scene import Scene


@dataclass(frozen=True)
class ReferenceGhost:
    entity: str
    physical_fields = frozenset()
    mdp_fields = ("reference",)

    @property
    def markers(self) -> tuple[RobotGhost, ...]:
        return (RobotGhost("Reference motion", self.entity),)

    def _initialize(self, scene: "Scene") -> "ReferenceGhost":
        return self

    def sample(self, physical_state: "TensorDict", mdp_state: "TensorDict") -> dict[str, "Tensor"]:
        return {"Reference motion": mdp_state["reference", "body_pose_w"]}
