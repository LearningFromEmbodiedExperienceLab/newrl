"""World-space terrain hits from a height-map sensor."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from tensordict import TensorDict
from torch import Tensor

from .markers import Marker

if TYPE_CHECKING:
    from newrl.scene import Scene


@dataclass(frozen=True)
class HeightScan:
    sensor: str
    size: float

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.sensor, "hit_pos_w"), (self.sensor, "valid")})

    @property
    def mdp_fields(self) -> tuple[str, ...]:
        return ()

    @property
    def markers(self) -> tuple[Marker, ...]:
        return (Marker(
            name=f"{self.sensor}/hits", kind="points", size=self.size,
            color=0x40FF80, group=f"Height scan ({self.sensor})",
        ),)

    def _initialize(self, scene: "Scene") -> Self:
        return self

    def sample(self, physical_state: TensorDict, mdp_state: TensorDict) -> dict[str, tuple[Tensor, Tensor]]:
        sensor = physical_state[self.sensor]
        return {
            f"{self.sensor}/hits": (
                sensor["hit_pos_w"].flatten(1, -2),
                sensor["valid"].flatten(1),
            ),
        }
