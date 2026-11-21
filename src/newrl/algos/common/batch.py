"""Rollout fields consumed by on-policy updates.

Time is dimension 1, so a collected batch has shape ``[num_envs, steps, ...]``.
The environment may still rename its own transition fields. ``train_onpolicy``
maps those fields onto this schema.

``policy_observation`` is the observation that produced the action.
``critic_observation`` is the value input at that same decision.
``next_critic_observation`` is the critic observation of the true successor,
including on termination or truncation, and it is recorded before any reset.
``terminated`` drops the value bootstrap. ``done`` is termination or truncation
and stops the advantage trace at the episode boundary.
"""

POLICY_OBSERVATION = "policy_observation"
CRITIC_OBSERVATION = "critic_observation"
NEXT_CRITIC_OBSERVATION = "next_critic_observation"
ACTION = "action"
ACTION_LOG_PROB = "action_log_prob"
REWARD = "reward"
TERMINATED = "terminated"
DONE = "done"

BATCH_KEYS = (
    POLICY_OBSERVATION,
    CRITIC_OBSERVATION,
    NEXT_CRITIC_OBSERVATION,
    ACTION,
    ACTION_LOG_PROB,
    REWARD,
    TERMINATED,
    DONE,
)
