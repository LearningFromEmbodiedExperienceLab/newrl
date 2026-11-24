"""Explicit physical dynamics with independently selected privileged inputs."""

import torch
from torch import nn
from tensordict import TensorDict

from newrl.algos.common.modules import MLP
from newrl.algos.common.normalization import EmpiricalNormalization


class ExplicitWorldModel(nn.Module):
    def __init__(self, representation, action_dim, hidden, observations, amp='none'):
        super().__init__()
        self.representation, self.observations, self.amp = representation, observations, amp
        self.state_norm = EmpiricalNormalization(representation.feature_dim, decay=1.0, eps=1e-4)
        self.action_norm = EmpiricalNormalization(action_dim, decay=1.0, eps=1e-4)
        width = 0 if observations is None else observations.shape.numel()
        self.observation_norm = EmpiricalNormalization(width, decay=1.0, eps=1e-4) if width else None
        self.network = nn.Sequential(
            MLP((representation.feature_dim + action_dim + width, *hidden), nn.SiLU,
                first_non_muon=True),
            nn.Linear(hidden[-1], representation.prediction_dim),
        )
        self.network[-1].weight._non_muon = True
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @torch.no_grad()
    def update_normalization(self, state, action, physical=None, memory=None):
        self.state_norm.update(self.representation.features(state))
        self.action_norm.update(action)
        if self.observation_norm is not None:
            physical = self.representation.decode(state) if physical is None else physical
            memory = TensorDict({}, state.batch_size) if memory is None else memory
            self.observation_norm.update(self.observations(physical, memory))

    def predict(self, state, action, physical, memory):
        """Compile dynamics, FP32 integration and successor FK as one boundary."""
        inputs = [self.state_norm.normalize(self.representation.features(state)),
                  self.action_norm.normalize(action)]
        if self.observation_norm is not None:
            memory = TensorDict({}, state.batch_size) if memory is None else memory
            inputs.append(self.observation_norm.normalize(self.observations(physical, memory)))
        with torch.autocast(action.device.type, dtype=torch.bfloat16,
                            enabled=self.amp == 'bf16' and action.is_cuda):
            hidden = self.network[0](torch.cat(inputs, -1))
        with torch.autocast(action.device.type, enabled=False):
            prediction = self.network[-1](hidden.float())
            state = self.representation.apply_prediction(state, prediction)
            physical = self.representation.decode(state)
        return state, physical

    def step(self, state, action, memory=None):
        physical = self.representation.decode(state)
        return self.predict(state, action, physical, memory)[0]

    def rollout(self, mdp, initial_state, memory, actions):
        """Open-loop predictions, scoring true successors against decision goals."""
        state = initial_state
        memory = mdp.restore_replay_state(memory)
        physical = self.representation.decode(state)
        states, rewards = [], []
        for t in range(actions.shape[-2]):
            state, physical = self.predict(state, actions[..., t, :], physical, memory)
            memory, transition = mdp.step(physical, memory, actions[..., t, :])
            states.append(state)
            values = [value.sum(-1, keepdim=True) for value in transition['reward'].values()]
            rewards.append(sum(values, actions.new_zeros((*initial_state.batch_size, 1))))
        dim = len(initial_state.batch_size)
        return TensorDict({'state': torch.stack(states, dim=dim), 'reward': torch.stack(rewards, dim=dim)}, batch_size=(*initial_state.batch_size, actions.shape[-2]))
