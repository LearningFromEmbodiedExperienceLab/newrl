"""Structural learner contracts for training dispatch and playback."""

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable, Self

from tensordict import TensorDict

if TYPE_CHECKING:
    from newrl.env import StatefulEnvironment


class Learner(Protocol):
    def act(self, tensordict: TensorDict, *, deterministic: bool = False) -> TensorDict: ...
    def training_state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> object: ...
    def eval(self) -> Self: ...


@runtime_checkable
class LearnerConfig(Protocol):
    algo_id: ClassVar[str]
    name: str

    def build(self, env: "StatefulEnvironment") -> Learner: ...


class TrainingLearner(Learner, Protocol):
    def record(self, transition: TensorDict) -> TensorDict: ...
    def collect(self, env: "StatefulEnvironment", steps: int, carry: TensorDict) -> tuple[TensorDict, TensorDict]: ...
    def train_op(self, batch: TensorDict) -> dict[str, float]: ...


@runtime_checkable
class TrainingConfig(LearnerConfig, Protocol):
    update_every: int

    def build(self, env: "StatefulEnvironment") -> TrainingLearner: ...
