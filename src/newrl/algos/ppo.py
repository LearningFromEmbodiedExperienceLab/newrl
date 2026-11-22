"""Clipped PPO over a collected rollout.

The batch schema lives in ``algos.common.batch``. This module does not step
the environment; ``scripts/train_onpolicy.py`` owns collection.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torch import nn

from .common.batch import (
    ACTION,
    ACTION_LOG_PROB,
    BATCH_KEYS,
    CRITIC_OBSERVATION,
    DONE,
    NEXT_CRITIC_OBSERVATION,
    POLICY_OBSERVATION,
    REWARD,
    TERMINATED,
)
from .common.modules import NormalActor, Value
from .common.normalization import EmpiricalNormalization, ReturnNormalization
from .common.optimizer import make_optimizer
from .common.policy_evaluation import generalized_advantage
from newrl.utils.config import cs  # noqa: E402
from newrl.utils.profiling import StageProfiler

_ACTIVATIONS = {
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


@dataclass
class PPOConfig:
    name: str = "ppo"
    update_every: int = 32
    num_epochs: int = 2
    num_minibatches: int = 8
    lr: float = 5e-4
    weight_decay: float = 0.01
    muon: bool = False
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float | tuple[float, float] = 0.2
    entropy_coef: float = 0.002
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    clamp_reward: bool = False
    normalize_reward: bool = False
    actor_hidden: tuple[int, ...] = (256, 256, 256)
    critic_hidden: tuple[int, ...] = (512, 256, 256)
    activation: str = "mish"
    min_std: float = 1e-2
    max_std: float = 1.0

    policy_obs_keys: tuple[str, ...] = ("proprio", "privileged", "command")
    critic_obs_keys: tuple[str, ...] = ("proprio", "privileged", "command")
    profile: bool = False

    def __post_init__(self):
        if self.update_every < 1 or self.num_epochs < 1 or self.num_minibatches < 1:
            raise ValueError(
                "update_every, num_epochs, and num_minibatches must be positive"
            )
        if not 0.0 <= self.gamma < 1.0 or not 0.0 <= self.lam <= 1.0:
            raise ValueError("gamma must be in [0, 1) and lam in [0, 1]")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {self.activation!r}; expected one of {tuple(_ACTIVATIONS)}"
            )
        clip_bounds(self.clip_eps)


def stack_groups(observation: TensorDict, groups: tuple[str, ...]) -> torch.Tensor:
    """Concatenate observation groups in the given order."""
    missing = [name for name in groups if name not in observation]
    if missing:
        raise KeyError(f"Observation is missing groups {missing}")
    return torch.cat([observation[name] for name in groups], dim=-1)


def clip_bounds(clip_eps: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(clip_eps, (int, float)):
        lower = upper = float(clip_eps)
    else:
        if len(clip_eps) != 2:
            raise ValueError(f"clip_eps must be a scalar or a pair, got {clip_eps!r}")
        lower, upper = float(clip_eps[0]), float(clip_eps[1])
    if lower < 0.0 or upper < 0.0:
        raise ValueError(
            f"clip_eps values must be non-negative, got {(lower, upper)!r}"
        )
    return lower, upper


cs.store("ppo", PPOConfig(), group="algo")

class PPO(nn.Module):
    def __init__(
        self,
        cfg: PPOConfig,
        *,
        policy_dim: int,
        critic_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        activation = _ACTIVATIONS[cfg.activation]
        self.clip_eps = clip_bounds(cfg.clip_eps)
        self.actor = NormalActor(
            policy_dim,
            action_dim,
            cfg.actor_hidden,
            activation,
            min_std=cfg.min_std,
            max_std=cfg.max_std,
        )
        self.critic = Value(critic_dim, cfg.critic_hidden, activation)
        self.policy_norm = EmpiricalNormalization(policy_dim)
        self.critic_norm = EmpiricalNormalization(critic_dim)
        self.return_norm = ReturnNormalization(cfg.gamma)
        self.optimizer = make_optimizer(
            self.actor,
            self.critic,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            muon=cfg.muon,
        )

    @property
    def device(self) -> torch.device:
        return self.policy_norm.sum.device

    def training_state_dict(self) -> dict:
        """Return model, normalizer, and optimizer state for a checkpoint."""
        return {"policy": self.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_training_state_dict(self, state: dict) -> None:
        """Restore a checkpoint into a PPO instance with the same configuration."""
        policy_state, optimizer_state = state["policy"], state["optimizer"]
        self.load_state_dict(policy_state)
        self.optimizer.load_state_dict(optimizer_state)

    @torch.no_grad()
    def act(self, tensordict: TensorDict, *, deterministic: bool = False) -> TensorDict:
        """Read observation groups and write the action and the stored features.

        Observation statistics update here. ``train_op`` freezes them.
        """
        observation = tensordict["observation"]
        policy_observation = stack_groups(observation, self.cfg.policy_obs_keys)
        critic_observation = stack_groups(observation, self.cfg.critic_obs_keys)
        normalized = self.policy_norm(policy_observation.to(self.device))
        distribution = self.actor.distribution(normalized)
        action = distribution.mean if deterministic else distribution.sample()
        tensordict[POLICY_OBSERVATION] = policy_observation
        tensordict[CRITIC_OBSERVATION] = critic_observation
        tensordict[ACTION] = action
        tensordict[ACTION_LOG_PROB] = distribution.log_prob(action).sum(
            -1, keepdim=True
        )
        return tensordict

    @torch.no_grad()
    def record(self, tensordict: TensorDict, transition: TensorDict) -> None:
        """Write the successor critic features from the true next observation."""
        tensordict[NEXT_CRITIC_OBSERVATION] = stack_groups(
            transition["observation"], self.cfg.critic_obs_keys
        ).detach()

    def train_op(self, batch: TensorDict) -> dict[str, float]:
        missing = [key for key in BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f"On-policy batch is missing {missing}")
        scalar_shape = (*batch.batch_size, 1)
        for key in (TERMINATED, DONE, ACTION_LOG_PROB):
            if batch[key].shape != scalar_shape:
                raise ValueError(
                    f"Expected {key!r} shape {scalar_shape}, got {tuple(batch[key].shape)}"
                )
        batch = batch.to(self.device)
        profiler = StageProfiler(self.device) if self.cfg.profile else None
        raw_reward = _aggregate_reward(batch)
        reward = self._scaled_reward(raw_reward, batch[DONE])
        self.critic_norm.update(
            torch.cat(
                [batch[CRITIC_OBSERVATION], batch[NEXT_CRITIC_OBSERVATION]], dim=1
            )
        )
        with self.policy_norm.freeze(), self.critic_norm.freeze():
            with torch.no_grad():
                value = self.critic(self.critic_norm(batch[CRITIC_OBSERVATION]))
                next_value = self.critic(
                    self.critic_norm(batch[NEXT_CRITIC_OBSERVATION])
                )
                advantage, returns = generalized_advantage(
                    reward,
                    batch[TERMINATED],
                    batch[DONE],
                    value,
                    next_value,
                    gamma=self.cfg.gamma,
                    lam=self.cfg.lam,
                )
                advantage_mean = advantage.mean()
                advantage_std = advantage.std().clamp_min(1e-7)
            normalized = (advantage - advantage_mean) / advantage_std
            training = TensorDict(
                {
                    POLICY_OBSERVATION: batch[POLICY_OBSERVATION],
                    CRITIC_OBSERVATION: batch[CRITIC_OBSERVATION],
                    ACTION: batch[ACTION],
                    ACTION_LOG_PROB: batch[ACTION_LOG_PROB],
                    "adv": normalized,
                    "ret": returns,
                },
                batch_size=batch.batch_size,
                device=self.device,
            )
            last_epoch: list[dict[str, float]] = []
            for _ in range(self.cfg.num_epochs):
                last_epoch = [
                    self._update(minibatch, profiler=profiler)
                    for minibatch in _minibatches(training, self.cfg.num_minibatches)
                ]

        metrics = _average(last_epoch)
        metrics["critic/neg_rew_ratio"] = float((raw_reward <= 0).float().mean())
        metrics["actor/lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["critic/adv_mean"] = float(advantage_mean)
        metrics["critic/adv_std"] = float(advantage_std)
        with torch.no_grad(), self.policy_norm.freeze(), self.critic_norm.freeze():
            new_log_prob = self.actor.log_prob(
                self.policy_norm(batch[POLICY_OBSERVATION]), batch[ACTION]
            )
            log_ratio = new_log_prob - batch[ACTION_LOG_PROB]
            metrics["actor/policy_gain"] = float((log_ratio * advantage).mean())
            predicted = self.critic(self.critic_norm(batch[CRITIC_OBSERVATION]))
            return_variance = returns.var().clamp_min(1e-7)
            metrics["critic/explained_var"] = float(
                1.0 - (predicted - returns).square().mean() / return_variance
            )
        if profiler is not None:
            for stage, (stream_ms, host_ms, calls) in profiler.totals().items():
                prefix = f"profile/ppo/{stage}"
                metrics[f"{prefix}/host_ms"] = host_ms
                metrics[f"{prefix}/calls"] = calls
                if self.device.type == "cuda":
                    metrics[f"{prefix}/stream_ms"] = stream_ms
        return dict(sorted(metrics.items()))

    def _scaled_reward(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Return rewards ready for GAE.

        Optional return normalization is applied to the raw reward. The factor
        ``(1 - gamma)`` is applied after that, once per call. Calling this with
        return normalization enabled advances the return statistics.
        """
        if self.cfg.clamp_reward:
            reward = reward.clamp_min(0.0)
        if self.cfg.normalize_reward:
            self.return_norm.update(reward, done)
            reward = reward / self.return_norm.scale()
        return reward * (1.0 - self.cfg.gamma)

    def _update(
        self, batch: TensorDict, *, profiler: StageProfiler | None = None
    ) -> dict[str, float]:
        stage = nullcontext if profiler is None else profiler.stage
        with stage("forward_loss"):
            observation = self.policy_norm(batch[POLICY_OBSERVATION])
            distribution = self.actor.distribution(observation)
            log_prob = distribution.log_prob(batch[ACTION]).sum(-1, keepdim=True)
            log_ratio = log_prob - batch[ACTION_LOG_PROB]
            ratio = log_ratio.exp()
            advantage = batch["adv"]
            lower, upper = self.clip_eps
            clipped = ratio.clamp(1.0 - lower, 1.0 + upper)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
            entropy = distribution.entropy().sum(-1, keepdim=True).mean()
            value = self.critic(self.critic_norm(batch[CRITIC_OBSERVATION]))
            value_loss = (value - batch["ret"]).square().mean()
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * entropy
            )
        with stage("backward"):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        with stage("grad_clip"):
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.max_grad_norm
            )
            critic_grad = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.max_grad_norm
            )
        with stage("optimizer_step"):
            self.optimizer.step()
        with stage("metrics"):
            detached_ratio = ratio.detach()
            detached_log_ratio = log_ratio.detach()
            return {
                "actor/policy_loss": float(policy_loss.detach()),
                "actor/entropy": float(entropy.detach()),
                "actor/grad_norm": float(actor_grad),
                "actor/approx_kl": float(
                    ((detached_ratio - 1.0) - detached_log_ratio).mean()
                ),
                "actor/clamp_pos": float((detached_ratio > 1.0 + upper).float().mean()),
                "actor/clamp_neg": float((detached_ratio < 1.0 - lower).float().mean()),
                "critic/value_loss": float(value_loss.detach()),
                "critic/value_mean": float(value.mean().detach()),
                "critic/grad_norm": float(critic_grad),
            }


def _aggregate_reward(batch: TensorDict) -> torch.Tensor:
    groups = batch[REWARD]
    scalar_shape = (*batch.batch_size, 1)
    if isinstance(groups, torch.Tensor):
        if groups.shape != scalar_shape:
            raise ValueError(
                f"Expected reward shape {scalar_shape}, got {tuple(groups.shape)}"
            )
        return groups
    parts = []
    for name, values in groups.items():
        if values.ndim != batch.ndim + 1 or values.shape[:-1] != batch.batch_size:
            raise ValueError(f"Expected one reward feature dimension for group {name!r}")
        parts.append(values)
    if not parts:
        return batch[DONE].new_zeros(scalar_shape, dtype=torch.float32)
    return torch.cat(parts, dim=-1).sum(dim=-1, keepdim=True)


def _minibatches(batch: TensorDict, num_minibatches: int):
    flat = batch.reshape(-1)
    usable = (flat.shape[0] // num_minibatches) * num_minibatches
    if usable == 0:
        raise ValueError(
            f"Batch of {flat.shape[0]} transitions is smaller than num_minibatches={num_minibatches}"
        )
    permutation = torch.randperm(usable, device=flat.device).reshape(
        num_minibatches, -1
    )
    for index in permutation:
        yield flat[index]


def _average(infos: list[dict[str, float]]) -> dict[str, float]:
    keys = infos[0]
    return {key: sum(info[key] for info in infos) / len(infos) for key in keys}
