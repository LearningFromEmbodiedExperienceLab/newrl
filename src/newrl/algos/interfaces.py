"""Structural contracts used by on-policy training and playback."""

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from tensordict import TensorDict

from .common.normalization import EmpiricalNormalization

if TYPE_CHECKING:
    from newrl.env import StatefulEnvironment


class OnPolicyLearner(Protocol):
    policy_norm: EmpiricalNormalization

    def act(self, tensordict: TensorDict, *, deterministic: bool = False) -> TensorDict: ...
    def record(self, tensordict: TensorDict, transition: TensorDict) -> None: ...
    def train_op(self, batch: TensorDict) -> dict[str, float]: ...
    def training_state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict): ...
    def eval(self): ...


@runtime_checkable
class OnPolicyConfig(Protocol):
    algo_id: ClassVar[str]
    name: str
    update_every: int

    def build(self, env: "StatefulEnvironment") -> OnPolicyLearner: ...
