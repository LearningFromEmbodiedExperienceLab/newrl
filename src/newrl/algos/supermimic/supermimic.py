"""Explicit-world-model SuperMimic: real replay, model fitting and imagined actor/critic updates."""

import copy
import math
from time import perf_counter
from newrl.algos.rollout import collect_rollout
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tensordict import TensorDict

from newrl.algos.common.batch import aggregate_reward, group_width, stack_groups
from newrl.algos.common.modules import MLP, Value
from newrl.algos.common.normalization import EmpiricalNormalization, ReturnNormalization
from newrl.algos.common.optimizer import build_optimizer
from newrl.algos.common.capture import CapturedUpdate
from newrl.mdp.base import ObservationGroup, MDP
from newrl.mdp.motion_tracking import MotionTrackingMDP
from newrl.mdp.observations import BodyPosition
from newrl.utils.config import cs
from .replay import ReplayBuffer
from .world_models import ExplicitWorldModel

from .representations import (
    GeneralizedCoordinates,
    GeneralizedCoordinatesConfig,
    MaximalCoordinates,
    MaximalCoordinatesConfig,
)

if TYPE_CHECKING:
    from newrl.env import StatefulEnvironment
    from newrl.configs import EnvironmentConfig


@dataclass
class SuperMimicConfig:
    # Registry identifier used to select the algorithm.
    algo_id: ClassVar[str] = 'supermimic'
    # Display name for runs and logs.
    name: str = 'supermimic'
    # Environment steps per world collected before each update.
    update_every: int = 4
    # Real steps per world collected before the first replay update.
    warmup_steps: int = 32
    # Replay capacity in control steps per world.
    buffer_steps: int = 1000
    # Replay windows sampled for each dynamics update.
    wm_batch: int = 2048
    # Number of transitions per autoregressive training window.
    wm_window: int = 8
    # Dynamics optimizer steps per collection cycle.
    wm_updates: int = 10
    # Replay states used to start each imagined rollout batch.
    imagination_batch: int = 4096
    # Number of imagined transitions before bootstrapping.
    horizon: int = 32
    # Hidden layer widths of the actor.
    actor_hidden: tuple[int, ...] = (512, 512, 512)
    # Hidden layer widths of the critic.
    critic_hidden: tuple[int, ...] = (512, 256, 256)
    # Hidden layer widths of the dynamics network.
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    # Number of independently trained value heads.
    critic_heads: int = 2
    # Passes over the imagined transitions.
    critic_epochs: int = 5
    # Critic minibatches per epoch.
    critic_minibatches: int = 4
    # Actor optimizer learning rate.
    actor_lr: float = 4e-4
    # Critic optimizer learning rate.
    critic_lr: float = 4e-4
    # Dynamics optimizer learning rate.
    wm_lr: float = 4e-4
    # Per-step reward discount.
    gamma: float = 0.99
    # Trace decay for imagined critic lambda returns.
    lam: float = 0.95
    # Standard deviation of Gaussian exploration noise in real collection.
    action_noise: float = 0.1
    # Maximum gradient norm for each optimizer update.
    grad_clip: float = 1.0
    # Elementwise actor gradient clamp after norm clipping. Zero disables it.
    actor_grad_clip_value: float = 0.1
    # Weight of analytic reward error in dynamics training.
    analytic_reward_loss_weight: float = 1.0
    # Use differentiable continuation weights when available in imagination.
    soft_survival: bool = True
    # Scale actor and critic rewards using statistics of real discounted returns.
    normalize_reward: bool = True
    # Optimizer selection for dynamics and actor.
    optim: str = 'muon_adam'
    # Optimizer selection for the critic ensemble.
    critic_optim: str = 'muon_adam'
    # CUDA hidden-layer precision. BF16 preserves FP32 output heads.
    amp: str = 'bf16'
    # Compile the CUDA dynamics prediction and forward kinematics.
    compile_world_model: bool = True
    # Capture and replay the CUDA dynamics optimizer update.
    capture_wm_update: bool = True
    # Capture and replay the CUDA actor and critic update.
    capture_policy_critic_update: bool = True
    # Additional physical observations supplied to the dynamics network.
    wm_observations: dict = field(default_factory=lambda: {
        'body_pos_b': BodyPosition('robot', (
            'pelvis', 'torso_link',
            '(left|right)_(hip_roll|knee|ankle_roll|shoulder_roll|elbow|wrist_yaw)_link',
        )),
    })
    # Physical state encoding and differentiable reconstruction for imagination.
    representation: GeneralizedCoordinatesConfig | MaximalCoordinatesConfig = field(
        default_factory=GeneralizedCoordinatesConfig
    )
    # Observation groups concatenated in order for the actor.
    policy_obs_keys: tuple[str, ...] = ('proprio', 'privileged', 'command')
    # Predict a correction to the next reference pose in joint target action units.
    residual_action: bool = True
    # Evaluation actor and observation normalizer EMA. Zero disables averaging.
    actor_ema: float = 0.995
    # Observation groups concatenated in order for the critic.
    critic_obs_keys: tuple[str, ...] = ('proprio', 'privileged', 'command')

    def __post_init__(self) -> None:
        if self.optim not in ('adam', 'adamw', 'muon', 'muon_adam') or self.critic_optim not in ('adam', 'adamw', 'muon', 'muon_adam'):
            raise ValueError('Unknown optimizer')
        if self.amp not in ('none', 'bf16'):
            raise ValueError('amp must be none or bf16')
        for name in ('update_every', 'warmup_steps', 'buffer_steps', 'wm_batch', 'wm_window', 'wm_updates', 'imagination_batch', 'horizon', 'critic_heads', 'critic_epochs', 'critic_minibatches'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.buffer_steps < self.wm_window or self.warmup_steps < self.wm_window:
            raise ValueError('Replay capacity and warmup must cover wm_window')
        if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1:
            raise ValueError('Invalid return discounts')
        if not 0 <= self.actor_ema < 1:
            raise ValueError('actor_ema must be in [0, 1)')
        for name in ('actor_lr', 'critic_lr', 'wm_lr', 'grad_clip'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('action_noise', 'actor_grad_clip_value', 'analytic_reward_loss_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for widths in (self.actor_hidden, self.critic_hidden, self.wm_hidden):
            if not widths or any(type(width) is not int or width < 1 for width in widths):
                raise ValueError('Network widths must be positive integers')

    def prepare_task(self, task: "EnvironmentConfig") -> "EnvironmentConfig":
        """Return an independent task with representation and replay outputs."""
        task = copy.deepcopy(task)
        task.output_physical_fields = tuple(dict.fromkeys((
            *task.output_physical_fields, *self.representation.physical_fields,
        )))
        task.output_mdp_fields = tuple(dict.fromkeys((
            *task.output_mdp_fields, *task.mdp.replay_fields,
            *(('future_reference',) if self.residual_action else ()),
        )))
        return task

    def build(self, env: "StatefulEnvironment") -> "SuperMimic":
        if env.config.randomization is not None:
            raise ValueError("SuperMimic requires fixed physical parameters until its dynamics model supports randomization context")
        representation = self.representation.build(env.scene)
        dynamics = ObservationGroup(self.wm_observations)._initialize(env.scene) if self.wm_observations else None
        if dynamics is not None:
            for _, term in dynamics.terms:
                required = getattr(term, 'physical_fields', None)
                if required is None or required - representation.physical_fields:
                    raise ValueError('World-model observations must declare reconstructable physical_fields')
                if getattr(term, 'state_history', {}):
                    raise ValueError('World-model observations must use current physical state, not term history')
        # Imagination evaluates only learner inputs, even if real playback keeps
        # additional diagnostic/camera observation groups enabled.
        mdp = copy.copy(env.mdp)
        if not self.policy_obs_keys or not self.critic_obs_keys:
            raise ValueError('Actor and critic observation groups must be nonempty')
        used = tuple(dict.fromkeys((*self.policy_obs_keys, *self.critic_obs_keys)))
        missing_groups = set(used) - set(mdp.observations)
        if missing_groups:
            raise ValueError(f'Missing learner observation groups: {sorted(missing_groups)}')
        mdp.observations = {name: mdp.observations[name] for name in used}
        terms = [term for group in (*mdp.observations.values(), *mdp.rewards.values()) for _, term in group.terms]
        terms.extend((*mdp.terminations.values(), *mdp.truncations.values()))
        histories = set().union(*(set(getattr(term, 'state_history', {})) for term in terms))
        mdp._history = {name: reader for name, reader in mdp._history.items() if name in histories}
        required = mdp.physical_fields
        if required is None:
            raise ValueError('SuperMimic requires explicit physical_fields on every MDP term')
        missing = required - representation.physical_fields
        if missing:
            raise ValueError(f'Representation cannot reconstruct task physical fields: {sorted(missing)}')
        if not set(self.representation.physical_fields) <= set(env.config.output_physical_fields):
            raise ValueError('Call prepare_task() before building the SuperMimic environment')
        if not set(env.mdp.replay_fields) <= set(env.output_mdp_fields):
            raise ValueError('SuperMimic environment is missing replay MDP outputs')
        from newrl.terrains import PlaneTerrainConfig
        if not isinstance(env.config.terrain, PlaneTerrainConfig) and env.config.terrain != 'plane':
            raise ValueError('SuperMimic currently requires plane terrain; nonuniform terrain needs dynamics context')
        action_mapping = None
        if self.residual_action:
            if not isinstance(mdp, MotionTrackingMDP) or set(env.scene.entities) != {mdp.entity}:
                raise ValueError('Residual actions require a single motion-tracking robot')
            if 'future_reference' not in env.output_mdp_fields:
                raise ValueError('Residual actions require future_reference in MDP outputs')
            robot = env.scene.entities[mdp.entity]
            if robot.velocity_mask is not None:
                raise ValueError('Residual actions require position-controlled joints')
            nominal = robot.default_position[0]
            scale = torch.as_tensor(robot.action.scale, device=nominal.device, dtype=nominal.dtype)
            if not (torch.isfinite(scale) & (scale > 0)).all():
                raise ValueError('Residual actions require finite, positive action scales')
            action_mapping = nominal.detach().clone(), scale.reciprocal()
        return SuperMimic(self, mdp, representation, env.device, env.action_dim, dynamics, action_mapping)


cs.store(SuperMimicConfig.algo_id, SuperMimicConfig(), group='algo')


@contextmanager
def _frozen(*modules: nn.Module) -> Iterator[None]:
    parameters = [(parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def lambda_returns(
    reward: torch.Tensor,
    continuation: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Detached critic targets; truncations retain bootstrap, falls remove it."""
    target = values[:, -1]
    targets = []
    for t in reversed(range(reward.shape[1])):
        target = reward[:, t] + gamma * continuation[:, t] * ((1 - lam) * values[:, t + 1] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1], 1)


class RewardNormalization(ReturnNormalization):
    """Reference return-STD scaling with a maximum-return safeguard.

    Statistics use raw real rewards; the bound comes from positive reward-term
    bounds, without timestep scaling. A nonpositive task bound falls back to one.
    """

    def __init__(self, gamma: float, max_step_reward: float) -> None:
        super().__init__(gamma)
        if not math.isfinite(max_step_reward) or max_step_reward <= 0:
            raise ValueError('Reward normalization requires a finite positive reward bound')
        self.max_return_bound = max_step_reward / (1 - gamma)
        self.register_buffer('sum', torch.zeros(()))
        self.register_buffer('max_return', torch.zeros(()))

    @torch.no_grad()
    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        previous = self.returns
        values = self._advance(reward, done)
        # Capture reads stable module storage across real collection calls.
        if previous.shape == self.returns.shape and previous.device == self.returns.device:
            previous.copy_(self.returns)
            self.returns = previous
        self.sum.add_(values.sum())
        self.square_sum.add_(values.square().sum())
        self.count.add_(values.numel())
        self.max_return.copy_(torch.maximum(self.max_return, values.abs().max()))

    def scale(self) -> torch.Tensor:
        count = self.count.clamp_min(1)
        variance = (self.square_sum / count - (self.sum / count).square()).clamp_min(0)
        return torch.maximum((variance + self.eps).sqrt(),
                             self.max_return / self.max_return_bound).clamp_min(self.eps).detach()


class SuperMimic(nn.Module):
    def __init__(
        self,
        config: SuperMimicConfig,
        mdp: MDP,
        representation: GeneralizedCoordinates | MaximalCoordinates,
        device: torch.device | str,
        action_dim: int,
        dynamics: ObservationGroup | None,
        action_mapping: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        super().__init__()
        self.config, self.mdp = config, mdp
        self.device = torch.device(device)
        if config.amp == 'bf16' and self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError('BF16 is unavailable on this GPU; use algo.amp=none')
        actor_width = group_width(mdp, config.policy_obs_keys)
        critic_width = group_width(mdp, config.critic_obs_keys)
        self.actor_norm = EmpiricalNormalization(actor_width, decay=1.0, eps=1e-4)
        self.critic_norm = EmpiricalNormalization(critic_width, decay=1.0, eps=1e-4)
        self.actor = nn.Sequential(MLP((actor_width, *config.actor_hidden), nn.SiLU, first_non_muon=True), nn.Linear(config.actor_hidden[-1], action_dim))
        self.actor[-1].weight._non_muon = True
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nominal, inverse_scale = action_mapping if action_mapping is not None else (None, None)
        self.register_buffer('_nominal_joint_pos', nominal, persistent=False)
        self.register_buffer('_inverse_action_scale', inverse_scale, persistent=False)
        self.actor_ema = (
            AveragedModel(
                nn.ModuleList([self.actor, self.actor_norm]),
                multi_avg_fn=get_ema_multi_avg_fn(config.actor_ema),
                use_buffers=True,
            ).requires_grad_(False)
            if config.actor_ema else None
        )
        self.critics = nn.ModuleList(Value(critic_width, config.critic_hidden, nn.SiLU) for _ in range(config.critic_heads))
        for critic in self.critics:
            nn.init.zeros_(critic.head.weight)
            nn.init.zeros_(critic.head.bias)
        self.wm = ExplicitWorldModel(representation, action_dim, config.wm_hidden, dynamics, config.amp)
        reward_bound = sum(max(term.range[1], 0.) for group in mdp.rewards.values() for _, term in group.terms)
        self.return_norm = RewardNormalization(config.gamma, reward_bound if config.normalize_reward and reward_bound > 0 else 1.)
        self.to(self.device)
        capture_wm = config.capture_wm_update and self.device.type == 'cuda'
        capture_policy = config.capture_policy_critic_update and self.device.type == 'cuda'
        self.opt_wm = build_optimizer(self.wm, lr=config.wm_lr, kind=config.optim, capturable=capture_wm)
        self.opt_actor = build_optimizer(self.actor, lr=config.actor_lr, kind=config.optim, adam_betas=(0.9, 0.95), capturable=capture_policy)
        self.opt_critic = build_optimizer(self.critics, lr=config.critic_lr, kind=config.critic_optim, capturable=capture_policy)
        # Replay world count is bound on first collection, not to imagination batch size.
        self.replay = None
        self.wm_steps = self.policy_steps = 0
        if config.compile_world_model and self.device.type == 'cuda':
            self.wm.predict = torch.compile(self.wm.predict, fullgraph=True, dynamic=False,
                                            options={'triton.cudagraphs': False})
        self._wm_update = CapturedUpdate(self._update_world_model_batch, self.wm, self.opt_wm, capture_wm)
        # Capture observes every weight/normalizer used by the imagined rollout.
        capture_modules = nn.ModuleList([self.actor, self.critics, self.wm, self.actor_norm, self.critic_norm, self.return_norm])
        self._policy_update = CapturedUpdate(self._update_policy_critic_batch, capture_modules,
                                            [self.opt_actor, self.opt_critic], capture_policy,
                                            label='policy/critic')

    @property
    def optimizers(self) -> tuple:
        return self.opt_wm, self.opt_actor, self.opt_critic

    @property
    def representation(self) -> GeneralizedCoordinates | MaximalCoordinates:
        return self.wm.representation

    def actor_action(
        self, observation: TensorDict, memory: TensorDict | None = None, *, use_ema: bool = False,
    ) -> torch.Tensor:
        actor, norm = self.actor, self.actor_norm
        if use_ema and self.actor_ema is not None:
            actor, norm = self.actor_ema.module
        features = norm.normalize(stack_groups(observation, self.config.policy_obs_keys))
        with torch.autocast(features.device.type, dtype=torch.bfloat16,
                            enabled=self.config.amp == 'bf16' and features.is_cuda):
            hidden = actor[0](features)
        with torch.autocast(features.device.type, enabled=False):
            action = actor[-1](hidden.float())
            if self._nominal_joint_pos is not None:
                if memory is None:
                    raise ValueError('Residual actions require decision-time MDP state')
                reference = memory['future_reference', 'joint_pos'][..., 0, :]
                action = action + (reference - self._nominal_joint_pos) * self._inverse_action_scale
            return action

    def value(self, observation: TensorDict) -> torch.Tensor:
        features = self.critic_norm.normalize(stack_groups(observation, self.config.critic_obs_keys))
        with torch.autocast(features.device.type, dtype=torch.bfloat16,
                            enabled=self.config.amp == 'bf16' and features.is_cuda):
            hidden = [critic.backbone(features) for critic in self.critics]
        with torch.autocast(features.device.type, enabled=False):
            return torch.cat([critic.head(value.float()) for critic, value in zip(self.critics, hidden, strict=True)], -1)

    @torch.no_grad()
    def act(self, tensordict: TensorDict, *, deterministic: bool = False) -> TensorDict:
        if self.training:
            self.actor_norm.update(stack_groups(tensordict['observation'], self.config.policy_obs_keys))
            self.critic_norm.update(stack_groups(tensordict['observation'], self.config.critic_obs_keys))
        action = self.actor_action(
            tensordict['observation'], tensordict.get('mdp_state'), use_ema=not self.training,
        )
        if not deterministic:
            action = action + self.config.action_noise * torch.randn_like(action)
        tensordict['action'] = action
        return tensordict

    @torch.no_grad()
    def warmup(self, env: "StatefulEnvironment", carry: TensorDict) -> tuple[TensorDict, TensorDict]:
        self.return_norm.returns.zero_()
        return self.collect(env, self.config.warmup_steps, carry)

    @torch.no_grad()
    def collect(self, env: "StatefulEnvironment", steps: int, carry: TensorDict) -> tuple[TensorDict, TensorDict]:
        if self.replay is None:
            self.replay = ReplayBuffer(self.config.buffer_steps, env.num_envs, self.device)
        batch, carry = collect_rollout(env, self, steps, carry)
        if self.config.normalize_reward:
            raw = aggregate_reward(batch)
            self.return_norm.update(raw, batch["next", "done"])
        return batch, carry

    @torch.no_grad()
    def record(self, transition: TensorDict) -> TensorDict:
        state = self.representation.encode(transition["physical_state"])
        action = transition["action"]
        self.wm.update_normalization(state, action, transition["physical_state"], transition["mdp_state"])
        successor = transition["next"]
        reward = aggregate_reward(transition)
        self.replay.add(TensorDict({
            "state": state,
            "mdp_state": transition["mdp_state"].select(*self.mdp.replay_fields),
            "action": action,
            "next": TensorDict({
                "state": self.representation.encode(successor["physical_state"]),
                "reward": reward,
                **{key: successor[key] for key in ("done", "terminated", "truncated")},
            }, transition.batch_size),
        }, transition.batch_size))
        return transition.select(("mdp_state", "episode_step"), *(("next", key) for key in ("reward", "terminated", "done", "termination_terms", "truncation_terms")))

    def train_op(self, batch: TensorDict) -> dict[str, float]:
        start = perf_counter()
        metrics = self.update_world_model()
        model_end = perf_counter()
        metrics.update(self.update_actor_critic())
        metrics.update({
            "replay/rows": self.replay.filled,
            "performance/world_model_time": model_end - start,
            "performance/policy_critic_time": perf_counter() - model_end,
        })
        return metrics

    def update_world_model(self) -> dict[str, float]:
        cfg = self.config
        rows = []
        for _ in range(cfg.wm_updates):
            batch = self.replay.sample_windows(cfg.wm_batch, cfg.wm_window)
            rows.append(self._wm_update(batch))
            self.wm_steps += 1
        return {key: value.item() for key, value in torch.stack(rows).mean(0).items()}

    def _update_world_model_batch(self, batch: TensorDict) -> TensorDict:
        cfg = self.config
        action = torch.where(batch['valid'], batch['action'], 0.)
        prediction = self.wm.rollout(self.mdp, batch['state'][:, 0], batch['mdp_state'][:, 0], action)
        state_loss = self.representation.compute_loss(prediction['state'], batch['next','state'], batch['valid'])
        reward_loss = ((prediction['reward'] - batch['next','reward']).abs() * batch['valid']).sum() / batch['valid'].sum().clamp_min(1)
        loss = state_loss + cfg.analytic_reward_loss_weight * reward_loss
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.wm.parameters(), cfg.grad_clip,
                                        error_if_nonfinite=not self._wm_update.enabled)
        if self._wm_update.enabled:
            torch._assert_async(torch.isfinite(norm), 'Non-finite WM gradient norm')
        self.opt_wm.step()
        return TensorDict({'wm/loss':loss.detach(), 'wm/state_loss':state_loss.detach(),
                           'wm/reward_loss':reward_loss.detach(), 'wm/grad_norm':norm.detach()}, [])

    def imagine(
        self,
        state: TensorDict,
        memory: TensorDict,
        horizon: int | None = None,
    ) -> TensorDict:
        horizon = self.config.horizon if horizon is None else horizon
        if horizon < 1:
            raise ValueError('Imagination horizon must be positive')
        memory = self.mdp.restore_replay_state(memory)
        physical = self.representation.decode(state)
        observation = self.mdp.compute_observation(physical, memory)
        observations, rewards, continuations = [observation], [], []
        for _ in range(horizon):
            action = self.actor_action(observation, memory)
            state, physical = self.wm.predict(state, action, physical, memory)
            memory, transition = self.mdp.step(physical, memory, action)
            observation = transition['observation']
            observations.append(observation)
            rewards.append(sum((value.sum(-1, keepdim=True) for value in transition['reward'].values()), action.new_zeros((*state.batch_size,1))))
            continuation = self.mdp.compute_survival(physical, memory) if self.config.soft_survival else (~transition['terminated']).float()
            continuations.append(continuation)
        return TensorDict({'observation':torch.stack(observations,1), 'reward':torch.stack(rewards,1), 'continuation':torch.stack(continuations,1)}, batch_size=state.batch_size)

    def update_actor_critic(self) -> dict[str, float]:
        cfg = self.config
        start = self.replay.sample_starts(cfg.imagination_batch, horizon=cfg.horizon, motion=getattr(self.mdp,'_motion',None))
        metrics = self._policy_update(start['state'], start['mdp_state'])
        if self.actor_ema is not None:
            self.actor_ema.update_parameters(nn.ModuleList([self.actor, self.actor_norm]))
        self.policy_steps += 1
        return {key: value.item() for key, value in metrics.items()}

    def _update_policy_critic_batch(self, state: TensorDict, memory: TensorDict) -> TensorDict:
        cfg = self.config
        capture = self._policy_update.enabled
        self.opt_wm.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        scale = self.return_norm.scale().reciprocal() if cfg.normalize_reward else state['root_pose'].new_ones(())
        with _frozen(self.wm, self.critics):
            imagined = self.imagine(state, memory)
            reward, continuation = imagined['reward'] * scale, imagined['continuation']
            alive = torch.ones_like(reward[:,0])
            returns = torch.zeros_like(alive)
            for t in range(cfg.horizon):
                returns = returns + cfg.gamma**t * alive * reward[:,t]
                alive = alive * continuation[:,t]
            bootstrap = self.value(imagined['observation'][:, -1]).mean(-1,keepdim=True)
            actor_loss = -(returns + cfg.gamma**cfg.horizon * alive * bootstrap).mean()
            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip, error_if_nonfinite=not capture)
            if capture:
                torch._assert_async(torch.isfinite(actor_norm), 'Non-finite actor gradient norm')
            if cfg.actor_grad_clip_value:
                nn.utils.clip_grad_value_(self.actor.parameters(), cfg.actor_grad_clip_value)
            self.opt_actor.step()
        with torch.no_grad():
            observations = imagined['observation'].detach()
            values = self.value(observations)
            targets = lambda_returns(reward.detach(), continuation.detach(), values, cfg.gamma, cfg.lam).flatten(0,1)
        inputs = observations[:, :-1].reshape(-1)
        critic_losses = []
        for _ in range(cfg.critic_epochs):
            for indices in torch.randperm(inputs.batch_size[0],device=self.device).tensor_split(min(cfg.critic_minibatches,len(inputs))):
                loss = (self.value(inputs[indices]) - targets[indices]).square().mean()
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.critics.parameters(), cfg.grad_clip, error_if_nonfinite=not capture)
                if capture:
                    torch._assert_async(torch.isfinite(norm), 'Non-finite critic gradient norm')
                self.opt_critic.step()
                critic_losses.append(loss.detach())
        return TensorDict({
            'actor/loss':actor_loss.detach(), 'actor/grad_norm':actor_norm.detach(),
            'critic/loss':torch.stack(critic_losses).mean(),
            'imagination/reward':imagined['reward'].detach().mean(),
            'imagination/survival':alive.detach().mean(), 'reward/scale':scale.detach(),
        }, [])

    def training_state_dict(self) -> dict[str, Any]:
        return {'policy':self.state_dict(), 'optimizers':{'world_model':self.opt_wm.state_dict(), 'actor':self.opt_actor.state_dict(), 'critic':self.opt_critic.state_dict()},
                'learner_updates':{'world_model':self.wm_steps,'policy':self.policy_steps}}

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state['policy'])
        for name, optimizer in (('world_model',self.opt_wm),('actor',self.opt_actor),('critic',self.opt_critic)):
            optimizer.load_state_dict(state['optimizers'][name])
        self.wm_steps, self.policy_steps = state['learner_updates']['world_model'], state['learner_updates']['policy']
        self._wm_update.reset()
        self._policy_update.reset()
