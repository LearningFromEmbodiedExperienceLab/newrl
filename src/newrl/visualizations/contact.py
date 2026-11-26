"""Last valid terrain touchdown positions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from tensordict import TensorDict
from torch import Tensor

from .markers import Marker

if TYPE_CHECKING:
    from newrl.scene import Scene


@dataclass(frozen=True)
class ContactPositions:
    sensor: str

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({
            (self.sensor, "last_contact_pos_w"),
            (self.sensor, "last_contact_pos_valid"),
        })

    @property
    def mdp_fields(self) -> tuple[str, ...]:
        return ()

    @property
    def markers(self) -> tuple[Marker, ...]:
        return (Marker(
            name=f"{self.sensor}/last_contact", kind="points",
            size=0.045, color=0xFF60C0,
            group=f"Last contacts ({self.sensor})",
        ),)

    def _initialize(self, scene: "Scene") -> Self:
        return self

    def sample(self, physical_state: TensorDict, mdp_state: TensorDict) -> dict[str, tuple[Tensor, Tensor]]:
        sensor = physical_state[self.sensor]
        return {
            f"{self.sensor}/last_contact": (
                sensor["last_contact_pos_w"],
                sensor["last_contact_pos_valid"],
            ),
        }
