"""Stack environment transitions into the on-policy batch schema.

The environment is not auto-reset. This collector records the successor
observation before resetting finished worlds, and the next action is taken from
the reset observation of those worlds.
"""

import torch
from tensordict import TensorDict

from .common.batch import (
    ACTION,
    ACTION_LOG_PROB,
    CRITIC_OBSERVATION,
    DONE,
    NEXT_CRITIC_OBSERVATION,
    POLICY_OBSERVATION,
    REWARD,
    TERMINATED,
)

# Fields currently produced by the environment. Keep this mapping here when
# those names change; the learner reads ``common.batch`` instead.
_ENV_POLICY = "policy_observation"
_ENV_CRITIC = "critic_observation"
_ENV_REWARD = "reward"
_ENV_TERMINATED = "terminated"
_ENV_DONE = "done"


def collect_rollout(
    env,
    policy,
    steps: int,
    carry=None # TODO: this should be a tensordict that is always not None
):
    """Collect ``steps`` transitions for every world.

    ``carry`` is the live ``(policy_observation, critic_observation)`` pair
    returned by the previous call. The first call resets every world.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if carry is None:
        transition = env.reset()
        policy_obs = _require(transition, _ENV_POLICY)
        critic_obs = _require(transition, _ENV_CRITIC)
    else:
        policy_obs, critic_obs = carry

    rows = []
    for _ in range(steps):
        # Clone before the step. Successor reads can alias simulator buffers that
        # the step is about to overwrite, and the stored decision must stay put.
        decision_policy = policy_obs.detach().clone()
        decision_critic = critic_obs.detach().clone()
        action, log_prob = policy.act(decision_policy)
        transition = env.step(action)
        successor_policy = _require(transition, _ENV_POLICY).detach().clone()
        successor_critic = _require(transition, _ENV_CRITIC).detach().clone()
        rows.append(
            {
                POLICY_OBSERVATION: decision_policy,
                CRITIC_OBSERVATION: decision_critic,
                NEXT_CRITIC_OBSERVATION: successor_critic,
                ACTION: action.detach(),
                ACTION_LOG_PROB: log_prob.detach(),
                REWARD: _require(transition, _ENV_REWARD).detach().clone(),
                TERMINATED: _require(transition, _ENV_TERMINATED).detach().clone(),
                DONE: _require(transition, _ENV_DONE).detach().clone(),
            }
        )
        finished = _done_index(transition[_ENV_DONE], env.num_envs)
        if finished.numel():
            reset = env.reset(finished)
            policy_obs = successor_policy.clone()
            critic_obs = successor_critic.clone()
            policy_obs[finished] = _require(reset, _ENV_POLICY)[finished]
            critic_obs[finished] = _require(reset, _ENV_CRITIC)[finished]
        else:
            policy_obs = successor_policy
            critic_obs = successor_critic

    stacked = {key: torch.stack([row[key] for row in rows], dim=1) for key in rows[0]}
    batch = TensorDict(stacked, batch_size=[env.num_envs, steps], device=stacked[REWARD].device)
    return batch, (policy_obs, critic_obs)


def _require(transition, key):
    if key not in transition:
        raise KeyError(
            f"Transition is missing {key!r}. The on-policy collector reads "
            "policy_observation, critic_observation, reward, terminated, and done."
        )
    return transition[key]


def _done_index(done: torch.Tensor, num_envs: int) -> torch.Tensor:
    return done.reshape(num_envs, -1)[:, 0].nonzero(as_tuple=True)[0]
