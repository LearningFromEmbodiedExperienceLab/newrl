"""Explicit-world-model SuperMimic: real replay, model fitting and imagined actor/critic updates."""

import copy
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import ClassVar
import math

import torch
from torch import nn
from tensordict import TensorDict

from newrl.algos.common.batch import group_width, stack_groups
from newrl.algos.common.modules import MLP, Value
from newrl.algos.common.normalization import EmpiricalNormalization, ReturnNormalization
from newrl.algos.common.optimizer import build_optimizer
from newrl.algos.common.capture import CapturedUpdate
from newrl.mdp.base import ObservationGroup
from newrl.mdp.observations import BodyPosition
from newrl.utils.config import cs
from .replay import ReplayBuffer
from .world_models import ExplicitWorldModel

from .representations import GeneralizedCoordinatesConfig


@dataclass
class SuperMimicConfig:
    algo_id: ClassVar[str] = 'supermimic'
    name: str = 'supermimic'
    update_every: int = 4
    warmup_steps: int = 32
    buffer_steps: int = 1000
    wm_batch: int = 2048
    wm_window: int = 8
    wm_updates: int = 10
    imagination_batch: int = 4096
    horizon: int = 32
    actor_hidden: tuple[int, ...] = (512, 512, 512)
    critic_hidden: tuple[int, ...] = (512, 256, 256)
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    critic_heads: int = 2
    critic_epochs: int = 5
    critic_minibatches: int = 4
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    wm_lr: float = 4e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    grad_clip: float = 1.0
    actor_grad_clip_value: float = 0.1
    analytic_reward_loss_weight: float = 1.0
    soft_survival: bool = True
    normalize_reward: bool = True
    optim: str = 'muon_adam'
    critic_optim: str = 'muon_adam'
    amp: str = 'bf16'
    compile_world_model: bool = True
    capture_wm_update: bool = True
    capture_policy_critic_update: bool = True
    wm_observations: dict = field(default_factory=lambda: {
        'body_pos_b': BodyPosition('robot', (
            'pelvis', 'torso_link',
            '(left|right)_(hip_roll|knee|ankle_roll|shoulder_roll|elbow|wrist_yaw)_link',
        )),
    })
    representation: GeneralizedCoordinatesConfig = field(default_factory=GeneralizedCoordinatesConfig)
    policy_obs_keys: tuple[str, ...] = ('proprio', 'privileged', 'command')
    critic_obs_keys: tuple[str, ...] = ('proprio', 'privileged', 'command')

    def __post_init__(self):
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
        for name in ('actor_lr', 'critic_lr', 'wm_lr', 'grad_clip'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('action_noise', 'actor_grad_clip_value', 'analytic_reward_loss_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for widths in (self.actor_hidden, self.critic_hidden, self.wm_hidden):
            if not widths or any(type(width) is not int or width < 1 for width in widths):
                raise ValueError('Network widths must be positive integers')

    def prepare_task(self, task):
        """Return an independent task with representation and replay outputs."""
        task = copy.deepcopy(task)
        task.output_physical_fields = tuple(dict.fromkeys((
            *task.output_physical_fields, *self.representation.physical_fields,
        )))
        task.output_mdp_fields = tuple(dict.fromkeys((
            *task.output_mdp_fields, *task.mdp.replay_fields,
        )))
        return task

    def build(self, env):
        representation = self.representation.build(env.scene)
        dynamics = ObservationGroup(self.wm_observations)._initialize(env.scene) if self.wm_observations else None
        if dynamics is not None:
            for term in dynamics.terms.values():
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
        terms = [term for group in (*mdp.observations.values(), *mdp.rewards.values()) for term in group.terms.values()]
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
        from newrl.terrain import PlaneTerrainConfig
        if not isinstance(env.config.terrain, PlaneTerrainConfig) and env.config.terrain != 'plane':
            raise ValueError('SuperMimic currently requires plane terrain; nonuniform terrain needs dynamics context')
        return SuperMimic(self, mdp, representation, env.device, env.action_dim, dynamics)


cs.store(SuperMimicConfig.algo_id, SuperMimicConfig(), group='algo')


@contextmanager
def _frozen(*modules):
    parameters = [(parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def lambda_returns(reward, continuation, values, gamma, lam):
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

    def __init__(self, gamma, max_step_reward):
        super().__init__(gamma)
        if not math.isfinite(max_step_reward) or max_step_reward <= 0:
            raise ValueError('Reward normalization requires a finite positive reward bound')
        self.max_return_bound = max_step_reward / (1 - gamma)
        self.register_buffer('sum', torch.zeros(()))
        self.register_buffer('max_return', torch.zeros(()))

    @torch.no_grad()
    def update(self, reward, done):
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

    def scale(self):
        count = self.count.clamp_min(1)
        variance = (self.square_sum / count - (self.sum / count).square()).clamp_min(0)
        return torch.maximum((variance + self.eps).sqrt(),
                             self.max_return / self.max_return_bound).clamp_min(self.eps).detach()


class SuperMimic(nn.Module):
    def __init__(self, config, mdp, representation, device, action_dim, dynamics):
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
        self.critics = nn.ModuleList(Value(critic_width, config.critic_hidden, nn.SiLU) for _ in range(config.critic_heads))
        for critic in self.critics:
            nn.init.zeros_(critic.head.weight)
            nn.init.zeros_(critic.head.bias)
        self.wm = ExplicitWorldModel(representation, action_dim, config.wm_hidden, dynamics, config.amp)
        reward_bound = sum(max(term.range[1], 0.) for group in mdp.rewards.values() for term in group.terms.values())
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
    def representation(self):
        return self.wm.representation

    def actor_action(self, observation):
        features = self.actor_norm.normalize(stack_groups(observation, self.config.policy_obs_keys))
        with torch.autocast(features.device.type, dtype=torch.bfloat16,
                            enabled=self.config.amp == 'bf16' and features.is_cuda):
            hidden = self.actor[0](features)
        with torch.autocast(features.device.type, enabled=False):
            return self.actor[-1](hidden.float())

    def value(self, observation):
        features = self.critic_norm.normalize(stack_groups(observation, self.config.critic_obs_keys))
        with torch.autocast(features.device.type, dtype=torch.bfloat16,
                            enabled=self.config.amp == 'bf16' and features.is_cuda):
            hidden = [critic.backbone(features) for critic in self.critics]
        with torch.autocast(features.device.type, enabled=False):
            return torch.cat([critic.head(value.float()) for critic, value in zip(self.critics, hidden, strict=True)], -1)

    @torch.no_grad()
    def act(self, tensordict, *, deterministic=False):
        action = self.actor_action(tensordict['observation'])
        if not deterministic:
            action = action + self.config.action_noise * torch.randn_like(action)
        tensordict['action'] = action
        return tensordict

    @torch.no_grad()
    def collect(self, env, steps, carry=None):
        if steps < 1:
            raise ValueError('Collection steps must be positive')
        if self.replay is None:
            self.replay = ReplayBuffer(self.config.buffer_steps, env.num_envs, self.device)
        if carry is None:
            self.return_norm.returns.zero_()
            carry = env.reset().select('observation', 'physical_state', 'mdp_state')
        rows = []
        for _ in range(steps):
            observation = carry['observation']
            self.actor_norm.update(stack_groups(observation, self.config.policy_obs_keys))
            self.critic_norm.update(stack_groups(observation, self.config.critic_obs_keys))
            state = self.representation.encode(carry['physical_state'])
            action = self.act(carry.select('observation').clone())['action']
            self.wm.update_normalization(state, action, carry['physical_state'], carry['mdp_state'])
            transition = env.step(action)
            reward = sum((values.sum(-1, keepdim=True) for values in transition['reward'].values()), action.new_zeros(env.num_envs, 1))
            self.replay.add(TensorDict({
                'state': state,
                'mdp_state': carry['mdp_state'].select(*self.mdp.replay_fields),
                'action': action,
                'next': TensorDict({
                    'state':self.representation.encode(transition['physical_state']),
                    'reward':reward,
                    'done':transition['done'],
                    'terminated':transition['terminated'],
                    'truncated':transition['truncated']
                }, [env.num_envs]),
            }, [env.num_envs]))
            rows.append(TensorDict({
                'mdp_state':carry['mdp_state'].select('episode_step').clone(),
                **{key:transition[key].detach().clone() for key in ('reward','terminated','done','termination_terms','truncation_terms')},
            }, [env.num_envs]))
            carry = transition.select('observation', 'physical_state', 'mdp_state')
            ended = transition['done'].squeeze(-1).nonzero(as_tuple=True)[0]
            if ended.numel():
                reset = env.reset(ended)
                carry[ended] = reset.select(*carry.keys())[ended]
        batch = torch.stack(rows, 1)
        if self.config.normalize_reward:
            raw = sum((values.sum(-1, keepdim=True) for values in batch['reward'].values()), batch['done'].new_zeros((*batch.batch_size,1), dtype=torch.float32))
            self.return_norm.update(raw, batch['done'])
        return batch, carry

    def update_world_model(self):
        cfg = self.config
        rows = []
        for _ in range(cfg.wm_updates):
            batch = self.replay.sample_windows(cfg.wm_batch, cfg.wm_window)
            rows.append(self._wm_update(batch))
            self.wm_steps += 1
        return {key: value.item() for key, value in torch.stack(rows).mean(0).items()}

    def _update_world_model_batch(self, batch):
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

    def imagine(self, state, memory, horizon=None):
        horizon = self.config.horizon if horizon is None else horizon
        if horizon < 1:
            raise ValueError('Imagination horizon must be positive')
        memory = self.mdp.restore_replay_state(memory)
        physical = self.representation.decode(state)
        observation = self.mdp.compute_observation(physical, memory)
        observations, rewards, continuations = [observation], [], []
        for _ in range(horizon):
            action = self.actor_action(observation)
            state, physical = self.wm.predict(state, action, physical, memory)
            memory, transition = self.mdp.step(physical, memory, action)
            observation = transition['observation']
            observations.append(observation)
            rewards.append(sum((value.sum(-1, keepdim=True) for value in transition['reward'].values()), action.new_zeros((*state.batch_size,1))))
            continuation = self.mdp.compute_survival(physical, memory) if self.config.soft_survival else (~transition['terminated']).float()
            continuations.append(continuation)
        return TensorDict({'observation':torch.stack(observations,1), 'reward':torch.stack(rewards,1), 'continuation':torch.stack(continuations,1)}, batch_size=state.batch_size)

    def update_actor_critic(self):
        cfg = self.config
        start = self.replay.sample_starts(cfg.imagination_batch, horizon=cfg.horizon, motion=getattr(self.mdp,'_motion',None))
        metrics = self._policy_update(start['state'], start['mdp_state'])
        self.policy_steps += 1
        return {key: value.item() for key, value in metrics.items()}

    def _update_policy_critic_batch(self, state, memory):
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

    def training_state_dict(self):
        return {'policy':self.state_dict(), 'optimizers':{'world_model':self.opt_wm.state_dict(), 'actor':self.opt_actor.state_dict(), 'critic':self.opt_critic.state_dict()},
                'learner_updates':{'world_model':self.wm_steps,'policy':self.policy_steps}}

    def load_training_state_dict(self, state):
        self.load_state_dict(state['policy'])
        for name, optimizer in (('world_model',self.opt_wm),('actor',self.opt_actor),('critic',self.opt_critic)):
            optimizer.load_state_dict(state['optimizers'][name])
        self.wm_steps, self.policy_steps = state['learner_updates']['world_model'], state['learner_updates']['policy']
        self._wm_update.reset()
        self._policy_update.reset()
