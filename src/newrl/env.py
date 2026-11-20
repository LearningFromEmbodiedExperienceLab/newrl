import torch
from tensordict import TensorDict
from typing import Optional


class Environment:
    def __init__(
        self,
    ):
        # unlike isaaclab and mjlab, we merge scene and sim
        # since it is really hard to fully decouple them.
        self.sim = ...
        self.mdp = ...
    
    def reset(self, inputs: TensorDict):
        env_ids = inputs.get("env_ids")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        else:
            env_ids = env_ids.to(self.device)
        # reset the simulation
        # reset the mdp
        return
    
    def step(self, inputs: TensorDict):
        # process and apply actions

        # step the simulation

        # step the mdp
        return