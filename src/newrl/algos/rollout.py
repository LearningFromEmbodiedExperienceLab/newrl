"""Stack environment transitions into a rollout batch.

The environment is not auto-reset. This collector asks the policy to write
whatever it wants into the step, then records the environment outcome. The
successor observation is kept for that step, and finished worlds are replaced
with their reset observation before the next action.
"""

import torch
from tensordict import TensorDict

_ENV_OBSERVATION = "observation"
_ENV_REWARD = "reward"
_ENV_TERMINATED = "terminated"
_ENV_DONE = "done"


def collect_rollout(env, policy, steps: int, carry: TensorDict):
    """Collect ``steps`` transitions for every world.

    ``carry`` is the TensorDict the policy reads for the next action. The
    policy writes its own buffer fields. The collector adds the reward and
    episode-end flags from the environment.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    rows = []
    for _ in range(steps):
        step = carry.clone()
        policy.act(step)
        transition = env.step(step["action"])
        policy.record(step, transition)
        step[_ENV_REWARD] = _require(transition, _ENV_REWARD).detach().clone()
        step[_ENV_TERMINATED] = _require(transition, _ENV_TERMINATED).detach().clone()
        step[_ENV_DONE] = _require(transition, _ENV_DONE).detach().clone()
        for field in ("termination_terms", "truncation_terms"):
            if field in transition:
                step[field] = transition[field].detach().clone()
        rows.append(step.exclude(_ENV_OBSERVATION))
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


def _require(transition, key):
    if key not in transition:
        raise KeyError(
            f"Transition is missing {key!r}. The collector reads "
            "observation, reward, terminated, and done."
        )
    return transition[key]
