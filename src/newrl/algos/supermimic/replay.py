"""Compact real transitions in a time/world ring; ending successors stay valid."""

import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, capacity, num_envs, device):
        if min(capacity, num_envs) < 1:
            raise ValueError('Replay capacity and world count must be positive')
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        self.storage = None
        self.ptr = self.filled = 0

    @torch.no_grad()
    def add(self, transition):
        if transition.batch_size != torch.Size([self.num_envs]):
            raise ValueError('Replay expects one real transition per world')
        if self.storage is None:
            self.storage = transition.detach().to(self.device).unsqueeze(0).expand(self.capacity, self.num_envs).apply(torch.empty_like)
        self.storage[self.ptr].copy_(transition.detach())
        self.ptr = (self.ptr + 1) % self.capacity
        self.filled = min(self.capacity, self.filled + 1)

    def sample_windows(self, batch_size, window):
        if batch_size < 1 or not 1 <= window <= self.filled:
            raise ValueError('Need positive batch/window sizes and enough replay rows')
        start = torch.randint(self.filled - window + 1, (batch_size,), device=self.device)
        world = torch.randint(self.num_envs, (batch_size,), device=self.device)
        offset = torch.arange(window, device=self.device)
        rows = (self.ptr - self.filled + start[:, None] + offset) % self.capacity
        batch = self.storage[rows, world[:, None]].clone()
        done = batch['next', 'done']
        prefix = torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), 1).int().cumprod(1).bool()
        count = batch['mdp_state', 'episode_step']
        contiguous = count == count[:, :1] + offset[None, :, None]
        batch['valid'] = prefix & contiguous.int().cumprod(1).bool()
        return batch

    def sample_starts(self, batch_size, *, horizon=0, motion=None):
        if batch_size < 1 or not self.filled:
            raise ValueError('Need a positive batch size and nonempty replay')
        rows = (torch.arange(self.filled, device=self.device) + self.ptr - self.filled) % self.capacity
        valid = torch.ones(self.filled, self.num_envs, dtype=torch.bool, device=self.device)
        if motion is not None and horizon:
            ids = self.storage['mdp_state', 'motion_id'][rows].squeeze(-1)
            frame = self.storage['mdp_state', 'motion_t'][rows].squeeze(-1)
            valid &= frame + horizon < motion.clip_end[ids] - motion.clip_start[ids]
        candidates = valid.flatten().nonzero(as_tuple=True)[0]
        if not candidates.numel():
            raise ValueError('No replay starts have enough remaining reference frames for imagination')
        chosen = candidates[torch.randint(len(candidates), (batch_size,), device=self.device)]
        return self.storage[rows[chosen // self.num_envs], chosen % self.num_envs].clone()
