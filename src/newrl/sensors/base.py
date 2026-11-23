"""Common interface for scene sensors and their configurations."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tensordict import TensorDict

if TYPE_CHECKING:
    import newton

    from ..scene import Scene


class Sensor(Protocol):
    """Expose named fields from the current simulator state."""

    @property
    def fields(self) -> tuple[str, ...]:
        """Names available to the scene's physical read plan."""
        ...

    def initialize(self, scene: "Scene") -> None:
        """Allocate resources after the environment selects active sensors."""
        ...

    def read(self, state: "newton.State", fields: tuple[str, ...]) -> TensorDict:
        """Return independent batched readings for the requested fields."""
        ...


@runtime_checkable
class SensorConfig(Protocol):
    """Build a sensor after the scene's model and entities are available."""

    def build(self, scene: "Scene") -> Sensor:
        """Bind the sensor to a scene."""
        ...
