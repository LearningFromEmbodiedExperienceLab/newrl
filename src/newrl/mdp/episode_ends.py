from dataclasses import dataclass, replace
from typing import ClassVar

from .observations import BodyHeight, ProjectedGravity
from .terms import StepFeatures


@dataclass(frozen=True)
class Timeout:
    steps: int
    physical_fields: ClassVar[frozenset] = frozenset()

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        return mdp_state["episode_step"].unsqueeze(-1) >= self.steps


@dataclass(frozen=True)
class Fall:
    height: BodyHeight
    gravity: ProjectedGravity
    min_height: float = 0.35
    min_upright_cosine: float = 0.5

    @property
    def physical_fields(self):
        return self.height.physical_fields | self.gravity.physical_fields

    def _initialize(self, scene):
        return replace(
            self,
            height=self.height._initialize(scene),
            gravity=self.gravity._initialize(scene),
        )

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        return self._check(
            self.height(physical_state, mdp_state, step_features=step_features),
            self.gravity(physical_state, mdp_state, step_features=step_features),
            mdp_state.get("spawn_height", 0.0),
        )

    def _check(self, height, gravity, spawn_height):
        return (height[:, :1] - spawn_height < self.min_height) | (
            -gravity[:, :1, 2] < self.min_upright_cosine
        )
