"""GenPO++ with a reversible, history-augmented action flow."""

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite, log, pi
from typing import ClassVar

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
    TERMINATED,
    aggregate_reward,
    group_width,
    minibatches,
    stack_groups,
)
from .common.modules import MLP, Value
from .common.normalization import EmpiricalNormalization, ReturnNormalization
from .common.optimizer import make_optimizer
from .common.policy_evaluation import clip_bounds, generalized_advantage
from newrl.utils.config import cs
from newrl.utils.profiling import StageProfiler

_ACTIVATIONS = {
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


FLOW_HIST = "flow_hist"


@dataclass
class GenPOConfig:
    algo_id: ClassVar[str] = "genpo"
    name: str = "genpo"
    update_every: int = 32
    num_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    weight_decay: float = 0.01
    muon: bool = False
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float | tuple[float, float] = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    clamp_reward: bool = False
    normalize_reward: bool = False
    actor_hidden: tuple[int, ...] = (256, 256, 256)
    critic_hidden: tuple[int, ...] = (512, 256, 256)
    activation: str = "mish"
    # Retained for config round trips with the earlier GenPO preset.
    min_std: float = 1e-2
    max_std: float = 1.0

    policy_obs_keys: tuple[str, ...] = ("proprio", "privileged", "command")
    critic_obs_keys: tuple[str, ...] = ("proprio", "privileged", "command")
    profile: bool = False
    timestep_embed_dim: int = 8
    flow_steps: int = 5
    sigma: float = 0.75
    desired_kl: float | None = 0.04

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
        if not isfinite(self.sigma) or self.sigma == 0:
            raise ValueError("GenPO sigma must be finite and nonzero")
        if self.flow_steps < 1:
            raise ValueError("GenPO flow_steps must be positive")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2:
            raise ValueError("GenPO timestep_embed_dim must be a positive even integer")
        if not self.actor_hidden or not self.critic_hidden:
            raise ValueError("GenPO actor_hidden and critic_hidden must be nonempty")
        if self.desired_kl is not None and (
            not isfinite(self.desired_kl) or self.desired_kl <= 0.0
        ):
            raise ValueError("GenPO desired_kl must be finite and positive, or None")

    def build(self, env) -> "GenPO":
        return GenPO(
            self,
            policy_dim=group_width(env.mdp, self.policy_obs_keys),
            critic_dim=group_width(env.mdp, self.critic_obs_keys),
            action_dim=env.action_dim,
        ).to(env.device)


def _base_log_prob(noise: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        noise.square().sum(dim=-1, keepdim=True) + noise.shape[-1] * log(2.0 * pi)
    )


class ReversibleFlowActor(nn.Module):
    """Integrate an augmented Gaussian and invert the resulting action/history pair."""

    def __init__(
        self,
        policy_dim: int,
        action_dim: int,
        activation: type[nn.Module],
        cfg: GenPOConfig,
    ):
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.flow_steps = cfg.flow_steps
        self.sigma = cfg.sigma
        self.dt = 1.0 / cfg.flow_steps
        self.backbone = MLP(
            (policy_dim + cfg.timestep_embed_dim + action_dim, *cfg.actor_hidden),
            activation=activation,
            first_non_muon=True,
        ).orth(gain=0.01)
        self.velocity_head = nn.Linear(cfg.actor_hidden[-1], action_dim)
        self.velocity_head.weight._non_muon = True
        nn.init.orthogonal_(self.velocity_head.weight, gain=0.01)
        nn.init.zeros_(self.velocity_head.bias)
        self.register_buffer(
            "time_freqs",
            2.0 ** torch.arange(cfg.timestep_embed_dim // 2, dtype=torch.float32),
            persistent=False,
        )

    def velocity(
        self, observation: torch.Tensor, x: torch.Tensor, step: int
    ) -> torch.Tensor:
        time = self.time_freqs.to(dtype=x.dtype) * (step * self.dt)
        embedding = torch.cat((time.cos(), time.sin())).expand(*x.shape[:-1], -1)
        features = torch.cat((observation, embedding, x), dim=-1)
        return self.velocity_head(self.backbone(features))

    def integrate(
        self, observation: torch.Tensor, x: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma
        for step in range(self.flow_steps):
            velocity = self.velocity(observation, x, step)
            next_x = (
                (1.0 - sigma) * x
                + sigma * history
                + (1.0 + sigma) * self.dt * velocity
            )
            x, history = next_x, x
        return x, history

    def invert(
        self, observation: torch.Tensor, action: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        sigma = self.sigma
        next_x, x = action, history
        for step in reversed(range(self.flow_steps)):
            velocity = self.velocity(observation, x, step)
            previous_x = (
                next_x - (1.0 - sigma) * x - (1.0 + sigma) * self.dt * velocity
            ) / sigma
            next_x, x = x, previous_x
        return torch.cat((next_x, x), dim=-1)

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*observation.shape[:-1], 2 * self.action_dim)
        noise = (
            observation.new_zeros(shape)
            if deterministic
            else torch.randn(shape, device=observation.device, dtype=observation.dtype)
        )
        x, history = noise.chunk(2, dim=-1)
        action, flow_hist = self.integrate(observation, x, history)
        return action, flow_hist, _base_log_prob(noise)

    def log_prob(
        self, observation: torch.Tensor, action: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        # The constant log determinant cancels in the PPO likelihood ratio.
        return _base_log_prob(self.invert(observation, action, history))


cs.store(GenPOConfig.algo_id, GenPOConfig(), group="algo")


class GenPO(nn.Module):
    def __init__(
        self,
        cfg: GenPOConfig,
        *,
        policy_dim: int,
        critic_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        activation = _ACTIVATIONS[cfg.activation]
        self.clip_eps = clip_bounds(cfg.clip_eps)
        self.actor = ReversibleFlowActor(policy_dim, action_dim, activation, cfg)
        self.augmented_entropy = (
            action_dim * (1.0 + log(2.0 * pi))
            + cfg.flow_steps * action_dim * log(abs(cfg.sigma))
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
        """Restore a checkpoint into a GenPO instance with the same configuration."""
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
        action, history, log_prob = self.actor.sample(
            normalized, deterministic=deterministic
        )
        tensordict[POLICY_OBSERVATION] = policy_observation
        tensordict[CRITIC_OBSERVATION] = critic_observation
        tensordict[ACTION] = action
        tensordict[FLOW_HIST] = history
        tensordict[ACTION_LOG_PROB] = log_prob
        return tensordict

    @torch.no_grad()
    def record(self, tensordict: TensorDict, transition: TensorDict) -> None:
        """Write the successor critic features from the true next observation."""
        tensordict[NEXT_CRITIC_OBSERVATION] = stack_groups(
            transition["observation"], self.cfg.critic_obs_keys
        ).detach()

    def train_op(self, batch: TensorDict) -> dict[str, float]:
        missing = [key for key in (*BATCH_KEYS, FLOW_HIST) if key not in batch]
        if missing:
            raise KeyError(f"On-policy batch is missing {missing}")
        scalar_shape = (*batch.batch_size, 1)
        for key in (TERMINATED, DONE, ACTION_LOG_PROB):
            if batch[key].shape != scalar_shape:
                raise ValueError(
                    f"Expected {key!r} shape {scalar_shape}, got {tuple(batch[key].shape)}"
                )
        if batch[FLOW_HIST].shape != batch[ACTION].shape:
            raise ValueError("flow_hist must match the action shape")
        batch = batch.to(self.device)
        profiler = StageProfiler(self.device) if self.cfg.profile else None
        raw_reward = aggregate_reward(batch)
        reward = raw_reward.clamp_min(0.0) if self.cfg.clamp_reward else raw_reward
        if self.cfg.normalize_reward:
            self.return_norm.update(reward, batch[DONE])
            reward = reward / self.return_norm.scale()
        reward = reward * (1.0 - self.cfg.gamma)
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
                    FLOW_HIST: batch[FLOW_HIST],
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
                    for minibatch in minibatches(training, self.cfg.num_minibatches)
                ]

        metrics = {
            key: sum(info[key] for info in last_epoch) / len(last_epoch)
            for key in last_epoch[0]
        }
        metrics["critic/neg_rew_ratio"] = float((raw_reward <= 0).float().mean())
        metrics["actor/lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["critic/adv_mean"] = float(advantage_mean)
        metrics["critic/adv_std"] = float(advantage_std)
        with torch.no_grad(), self.policy_norm.freeze(), self.critic_norm.freeze():
            new_log_prob = self.actor.log_prob(
                self.policy_norm(batch[POLICY_OBSERVATION]),
                batch[ACTION],
                batch[FLOW_HIST],
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
                prefix = f"profile/genpo/{stage}"
                metrics[f"{prefix}/host_ms"] = host_ms
                metrics[f"{prefix}/calls"] = calls
                if self.device.type == "cuda":
                    metrics[f"{prefix}/stream_ms"] = stream_ms
        return dict(sorted(metrics.items()))

    def _update(
        self, batch: TensorDict, *, profiler: StageProfiler | None = None
    ) -> dict[str, float]:
        stage = nullcontext if profiler is None else profiler.stage
        with stage("forward_loss"):
            observation = self.policy_norm(batch[POLICY_OBSERVATION])
            log_prob = self.actor.log_prob(observation, batch[ACTION], batch[FLOW_HIST])
            log_ratio = log_prob - batch[ACTION_LOG_PROB]
            ratio = log_ratio.exp()
            advantage = batch["adv"]
            lower, upper = self.clip_eps
            clipped = ratio.clamp(1.0 - lower, 1.0 + upper)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
            value = self.critic(self.critic_norm(batch[CRITIC_OBSERVATION]))
            value_loss = (value - batch["ret"]).square().mean()
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * self.augmented_entropy
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
            augmented_kl = float(-detached_log_ratio.mean())
            target = self.cfg.desired_kl
            if target is not None:
                factor = 1.0
                if augmented_kl > 2.0 * target:
                    factor = 1.0 / 1.5
                elif 0.0 < augmented_kl < 0.5 * target:
                    factor = 1.5
                if factor != 1.0:
                    # AdamW and Muon keep separate actor groups. Use the live groups
                    # so this also works after loading the optimizer checkpoint.
                    actor_params = {id(param) for param in self.actor.parameters()}
                    for group in self.optimizer.param_groups:
                        if group["params"] and id(group["params"][0]) in actor_params:
                            group["lr"] = min(1e-3, max(1e-5, group["lr"] * factor))
            return {
                "actor/policy_loss": float(policy_loss.detach()),
                "actor/entropy": self.augmented_entropy,
                "actor/aug_kl": augmented_kl,
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


