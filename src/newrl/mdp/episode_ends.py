from dataclasses import dataclass, replace
from typing import ClassVar

from .observations import BodyHeight, ProjectedGravity


@dataclass(frozen=True)
class Timeout:
    steps: int
    physical_fields: ClassVar[frozenset] = frozenset()

    def __call__(self, physical_state, mdp_state):
        return mdp_state["episode_step"] >= self.steps


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


@dataclass(frozen=True)
class ClipEnd:
    """Truncate at the last reference frame without crossing clip boundaries."""

    physical_fields: ClassVar[frozenset] = frozenset()

    def __call__(self, physical_state, mdp_state):
        return mdp_state["clip_end"]


@dataclass(frozen=True)
class TrackingDeviation:
    """Terminate on reference-relative root height or projected-gravity error."""

    body: BodyHeight
    quantity: str = "height"
    threshold: float = 0.25
    softness: float = 0.02

    def __post_init__(self):
        if self.quantity not in ("height", "tilt") or self.threshold <= 0 or self.softness <= 0:
            raise ValueError("Tracking deviation requires height/tilt and a positive threshold")

    @property
    def physical_fields(self):
        return self.body.physical_fields

    def _initialize(self, scene):
        return replace(self, body=self.body._initialize(scene))

    def error(self, physical_state, mdp_state):
        pose = physical_state[self.body.entity, "body_pose_w"].index_select(-2, self.body._indices)[..., 0, :]
        target = mdp_state["reference", "root_pose"]
        if self.quantity == "height":
            error = (pose[..., 2:3] - target[..., 2:3]).abs()
        else:
            actual_z = 2 * pose[..., 3:5].square().sum(-1, keepdim=True) - 1
            target_z = 2 * target[..., 3:5].square().sum(-1, keepdim=True) - 1
            error = (actual_z - target_z).abs()
        return error

    def __call__(self, physical_state, mdp_state):
        return self.error(physical_state, mdp_state) > self.threshold

    def survival(self, physical_state, mdp_state):
        return ((self.threshold - self.error(physical_state, mdp_state)) / self.softness).sigmoid()
