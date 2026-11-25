"""Clipped PPO with mirror-equivariant Gaussian policy regularization.

The batch schema lives in ``algos.common.batch``. The trainer owns collection.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from .common.batch import (
    ACTION,
    ACTION_LOG_PROB,
    BATCH_KEYS,
    CRITIC_OBSERVATION,
    DONE,
    NEXT_CRITIC_OBSERVATION,
    POLICY_OBSERVATION,
    TERMINATED,
    aggregate_reward,
    collect_metrics,
    group_width,
    minibatches,
    stack_groups,
)
from .common.modules import NormalActor, Value
from .common.normalization import EmpiricalNormalization, ReturnNormalization
from .common.optimizer import make_optimizer
from .common.policy_evaluation import clip_bounds, generalized_advantage
from newrl.utils.config import cs  # noqa: E402
from newrl.utils.profiling import StageProfiler
from newrl.utils.distributed import average_gradients, average_metrics, world_size
from newrl.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    from newrl.env import StatefulEnvironment

_ACTIVATIONS = {
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


@dataclass
class PPOSymAugConfig:
    # Registry identifier used to select the algorithm.
    algo_id: ClassVar[str] = "ppo_symaug"
    # Display name for runs and logs.
    name: str = "ppo_symaug"
    # Environment steps per world collected before each update.
    update_every: int = 32
    # Passes over the rollout per update.
    num_epochs: int = 2
    # Minibatches per epoch.
    num_minibatches: int = 8
    # Learning rate for the actor and critic optimizer.
    lr: float = 5e-4
    # Optimizer weight decay.
    weight_decay: float = 0.01
    # Use Muon for eligible matrices and AdamW for the remaining parameters.
    muon: bool = False
    # Per-step reward discount.
    gamma: float = 0.99
    # Generalized advantage estimation trace decay.
    lam: float = 0.95
    # Ratio clip around one, as a symmetric width or (lower, upper) widths.
    clip_eps: float | tuple[float, float] = 0.2
    # Coefficient of the entropy bonus in the actor loss.
    entropy_coef: float = 0.002
    # Weight of the critic regression loss.
    value_coef: float = 1.0
    # Maximum gradient norm, applied separately to actor and critic.
    max_grad_norm: float = 1.0
    # Floor the summed per-step reward at zero before computing returns.
    clamp_reward: bool = False
    # Scale rewards by the RMS of real discounted returns.
    normalize_reward: bool = False
    # Hidden layer widths of the actor.
    actor_hidden: tuple[int, ...] = (512, 512, 256)
    # Hidden layer widths of the critic.
    critic_hidden: tuple[int, ...] = (512, 512, 256)
    # Hidden layer activation name.
    activation: str = "mish"
    # Lower bound on the Gaussian standard deviation.
    min_std: float = 1e-2
    # Upper bound on the Gaussian standard deviation.
    max_std: float = 1.0

    # Observation groups concatenated in order for the actor.
    policy_obs_keys: tuple[str, ...] = ("proprio", "privileged", "command")
    # Observation groups concatenated in order for the critic.
    critic_obs_keys: tuple[str, ...] = ("proprio", "privileged", "command")
    # Collect timing metrics for the learner update.
    profile: bool = False
    # Weight of the mirror-equivariance loss.
    symmetry_coef: float = 1.0

    def __post_init__(self) -> None:
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
        if not isfinite(self.symmetry_coef) or self.symmetry_coef <= 0.0:
            raise ValueError("symmetry_coef must be finite and positive")

    def build(self, env: "StatefulEnvironment") -> "PPOSymAug":
        return PPOSymAug(
            self,
            policy_dim=group_width(env.mdp, self.policy_obs_keys),
            critic_dim=group_width(env.mdp, self.critic_obs_keys),
            action_dim=env.action_dim,
            observation_mirror=env.observation_symmetry_transform(self.policy_obs_keys),
            action_mirror=env.scene.action_symmetry_transform(),
        ).to(env.device)


cs.store(PPOSymAugConfig.algo_id, PPOSymAugConfig(), group="algo")


class PPOSymAug(nn.Module):
    def __init__(
        self,
        cfg: PPOSymAugConfig,
        *,
        policy_dim: int,
        critic_dim: int,
        action_dim: int,
        observation_mirror: SymmetryTransform,
        action_mirror: SymmetryTransform,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.observation_mirror = observation_mirror
        self.action_mirror = action_mirror
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
        self._rollout_moments: tuple[torch.Tensor, ...] | None = None
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

    def begin_rollout(self) -> None:
        if world_size() > 1:
            norm = self.policy_norm
            self._rollout_moments = tuple(v.clone() for v in (norm.sum, norm.sum_sq, norm.count))

    def training_state_dict(self) -> dict:
        """Return model, normalizer, and optimizer state for a checkpoint."""
        return {"policy": self.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_training_state_dict(self, state: dict) -> None:
        """Restore a checkpoint into a PPOSymAug instance with the same configuration."""
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
        action = distribution.mean if deterministic else distribution.rsample()
        tensordict[POLICY_OBSERVATION] = normalized
        tensordict["mirrored_policy_observation"] = self.policy_norm.normalize(
            self.observation_mirror(policy_observation.to(self.device))
        )
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
        if world_size() > 1:
            if self._rollout_moments is None:
                raise RuntimeError("Call begin_rollout before distributed collection")
            self.policy_norm.merge(self._rollout_moments, batch.batch_size[1])
            self._rollout_moments = None
        profiler = StageProfiler(self.device) if self.cfg.profile else None
        raw_reward = aggregate_reward(batch)
        reward = raw_reward.clamp_min(0.0) if self.cfg.clamp_reward else raw_reward
        if self.cfg.normalize_reward:
            self.return_norm.update(reward, batch[DONE], distributed=True)
            reward = reward / self.return_norm.scale()
        reward = reward * (1.0 - self.cfg.gamma)
        self.critic_norm.update(
            torch.cat(
                [batch[CRITIC_OBSERVATION], batch[NEXT_CRITIC_OBSERVATION]], dim=1
            ), distributed=True
        )
        with self.critic_norm.freeze():
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
            last_epoch: list[dict[str, torch.Tensor]] = []
            training["mirrored_policy_observation"] = batch["mirrored_policy_observation"]
            for _ in range(self.cfg.num_epochs):
                last_epoch = [
                    self._update(minibatch, profiler=profiler)
                    for minibatch in minibatches(training, self.cfg.num_minibatches)
                ]

        rollout_metrics = {
            "critic/neg_rew_ratio": (raw_reward <= 0).float().mean(),
            "critic/adv_mean": advantage_mean,
            "critic/adv_std": advantage_std,
        }
        with torch.no_grad(), self.critic_norm.freeze():
            new_log_prob = self.actor.log_prob(
                batch[POLICY_OBSERVATION], batch[ACTION]
            )
            log_ratio = new_log_prob - batch[ACTION_LOG_PROB]
            rollout_metrics["actor/policy_gain"] = (log_ratio * advantage).mean()
            predicted = self.critic(self.critic_norm(batch[CRITIC_OBSERVATION]))
            return_variance = returns.var().clamp_min(1e-7)
            rollout_metrics["critic/explained_var"] = (
                1.0 - (predicted - returns).square().mean() / return_variance
            )
        metrics = average_metrics(collect_metrics(last_epoch, rollout_metrics), self.device)
        metrics["actor/lr"] = self.optimizer.param_groups[0]["lr"]
        if profiler is not None:
            # The metric transfer above already waited for this stream.
            for stage, (stream_ms, host_ms, calls) in profiler.totals(synchronize=False).items():
                prefix = f"profile/ppo_symaug/{stage}"
                metrics[f"{prefix}/host_ms"] = host_ms
                metrics[f"{prefix}/calls"] = calls
                if self.device.type == "cuda":
                    metrics[f"{prefix}/stream_ms"] = stream_ms
        return dict(sorted(metrics.items()))

    def _update(
        self, batch: TensorDict, *, profiler: StageProfiler | None = None
    ) -> dict[str, torch.Tensor]:
        stage = nullcontext if profiler is None else profiler.stage
        with stage("forward_loss"):
            observation = batch[POLICY_OBSERVATION]
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
            mirrored_observation = batch["mirrored_policy_observation"]
            mirrored_mean = self.actor.mean(self.actor.backbone(mirrored_observation))
            scale = self.actor.scale()
            symmetry_loss = F.mse_loss(
                mirrored_mean, self.action_mirror(distribution.mean)
            ) + F.mse_loss(scale, self.action_mirror.permute(scale))
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * entropy
                + self.cfg.symmetry_coef * symmetry_loss
            )
        with stage("backward"):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        with stage("gradient_sync"):
            average_gradients(self.parameters())
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
                "actor/policy_loss": policy_loss.detach(),
                "actor/symmetry_loss": symmetry_loss.detach(),
                "actor/entropy": entropy.detach(),
                "actor/grad_norm": actor_grad,
                "actor/approx_kl": ((detached_ratio - 1.0) - detached_log_ratio).mean(),
                "actor/clamp_pos": (detached_ratio > 1.0 + upper).float().mean(),
                "actor/clamp_neg": (detached_ratio < 1.0 - lower).float().mean(),
                "critic/value_loss": value_loss.detach(),
                "critic/value_mean": value.mean().detach(),
                "critic/grad_norm": critic_grad,
            }
