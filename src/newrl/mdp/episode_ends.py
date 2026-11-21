from dataclasses import dataclass, replace
from typing import ClassVar

from .observations import BodyHeight, ProjectedGravity


@dataclass(frozen=True)
class Timeout:
    steps: int
    physical_fields: ClassVar[frozenset] = frozenset()

    def __call__(self, physical_state, mdp_state):
        return mdp_state["episode_step"].unsqueeze(-1) >= self.steps


@dataclass(frozen=True)
class HeightTermination:
    height: BodyHeight
    min_height: float = 0.35

    @property
    def physical_fields(self):
        return self.height.physical_fields

    def _initialize(self, scene):
        return replace(self, height=self.height._initialize(scene))

    def __call__(self, physical_state, mdp_state):
        height = physical_state[self.height.entity, "body_pose_w"].index_select(
            1, self.height._indices
        )[..., 2]
        return height[:, :1] - mdp_state.get("spawn_height", 0.0) < self.min_height


@dataclass(frozen=True)
class TiltTermination:
    gravity: ProjectedGravity
    min_upright_cosine: float = 0.5

    @property
    def physical_fields(self):
        return self.gravity.physical_fields

    def _initialize(self, scene):
        return replace(self, gravity=self.gravity._initialize(scene))

    def __call__(self, physical_state, mdp_state):
        gravity = self.gravity(physical_state, mdp_state)
        return -gravity[:, :1, 2] < self.min_upright_cosine
