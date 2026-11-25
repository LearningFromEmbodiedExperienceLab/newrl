"""Stack environment transitions into a rollout batch.

The environment is not auto-reset. This collector asks the policy to write
whatever it wants into the step, then records the environment outcome. The
successor observation is kept for that step, and finished worlds are replaced
with their reset observation before the next action.
"""

from typing import TYPE_CHECKING

from newrl.algos.interfaces import TrainingLearner

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from newrl.env import StatefulEnvironment

_ENV_OBSERVATION = "observation"
_ENV_DONE = "done"


def collect_rollout(
    env: "StatefulEnvironment",
    policy: TrainingLearner,
    steps: int,
    carry: TensorDict,
) -> tuple[TensorDict, TensorDict]:
    """Collect ``steps`` transitions for every world.

    ``carry`` is the TensorDict the policy reads for the next action. The
    policy writes its own buffer fields. The collector attaches the true
    successor under "next" and lets the learner choose the retained row.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    carry = _output_carry(carry)
    rows = []
    for _ in range(steps):
        step = policy.act(carry.clone())
        transition = env.step(step)
        step["next"] = transition
        rows.append(policy.record(step))
        carry = _output_carry(transition)
        finished = transition[_ENV_DONE].squeeze(-1).nonzero(as_tuple=True)[0]
        if finished.numel():
            reset = env.reset(finished)
            carry[finished] = reset.select(*carry.keys())[finished]
    return torch.stack(rows, dim=1), carry


def _output_carry(transition: TensorDict) -> TensorDict:
    _require(transition, _ENV_OBSERVATION)
    fields = [
        name
        for name in (_ENV_OBSERVATION, "mdp_state", "physical_state")
        if name in transition
    ]
    return transition.select(*fields).detach().clone()


def _require(transition: TensorDict, key: str) -> torch.Tensor | TensorDict:
    if key not in transition:
        raise KeyError(
            f"Transition is missing {key!r}. The collector reads "
            "observation, reward, terminated, and done."
        )
    return transition[key]
