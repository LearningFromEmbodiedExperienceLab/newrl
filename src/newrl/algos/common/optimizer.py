"""PPO optimizers with an optional Muon update for hidden weight matrices."""

import torch
from torch import nn


class MuonAdamW:
    """Step the two disjoint parameter sets together."""

    def __init__(self, adamw: torch.optim.AdamW, muon: torch.optim.Muon):
        self.adamw = adamw
        self.muon = muon

    @property
    def param_groups(self):
        return [*self.adamw.param_groups, *self.muon.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.adamw.step()
        self.muon.step()

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict) -> None:
        adamw_state, muon_state = state_dict["adamw"], state_dict["muon"]
        self.adamw.load_state_dict(adamw_state)
        self.muon.load_state_dict(muon_state)


def make_optimizer(
    actor: nn.Module,
    critic: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    muon: bool,
) -> torch.optim.AdamW | MuonAdamW:
    adamw_groups = []
    muon_groups = []
    for module in (actor, critic):
        adamw_params = []
        muon_params = []
        for parameter in module.parameters():
            if (
                muon
                and parameter.ndim == 2
                and not getattr(parameter, "_non_muon", False)
            ):
                muon_params.append(parameter)
            else:
                adamw_params.append(parameter)
        adamw_groups.append({"params": adamw_params})
        if muon_params:
            muon_groups.append({"params": muon_params})

    adamw = torch.optim.AdamW(adamw_groups, lr=lr, weight_decay=weight_decay)
    if not muon_groups:
        return adamw
    return MuonAdamW(
        adamw,
        torch.optim.Muon(
            muon_groups,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        ),
    )
