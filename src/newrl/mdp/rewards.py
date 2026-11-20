from dataclasses import dataclass, replace

from .observations import BodyAngularVelocity, BodyLinearVelocity
from .terms import StepFeatures


@dataclass(frozen=True)
class ActionRate:
    """Penalize squared changes in normalized action per environment."""

    weight: float

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        delta = mdp_state["action"] - mdp_state["previous_action"]
        return -self.weight * delta.square().sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class TrackVelocity:
    """Exponentially reward matching a command component in the yaw frame."""

    velocity: BodyLinearVelocity | BodyAngularVelocity
    components: slice
    weight: float
    sigma: float

    def _initialize(self, scene):
        return replace(self, velocity=self.velocity._initialize(scene))

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        measured = self.velocity(
            physical_state, mdp_state, step_features=step_features
        )[:, 0, self.components]
        target = mdp_state["velocity_command"][:, self.components]
        squared_error = (measured - target).square().sum(dim=-1, keepdim=True)
        return self.weight * (-squared_error / self.sigma**2).exp()
