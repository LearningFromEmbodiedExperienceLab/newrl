"""Optimizers with an optional Muon update for hidden weight matrices."""

from torch.optim import Optimizer

from typing import Any

import torch
from torch import nn


class MuonAdamW:
    """Step the two disjoint parameter sets together."""

    def __init__(self, adamw: torch.optim.Optimizer, muon: torch.optim.Muon) -> None:
        self.adamw = adamw
        self.muon = muon

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Muon]:
        return (self.adamw, self.muon)

    @property
    def state(self) -> dict[torch.Tensor, dict[str, Any]]:
        return {**self.adamw.state, **self.muon.state}

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [*self.adamw.param_groups, *self.muon.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.adamw.step()
        self.muon.step()

    def state_dict(self) -> dict[str, Any]:
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
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


def build_optimizer(
    module: nn.Module,
    *,
    lr: float,
    kind: str,
    adam_betas: tuple[float, float]=(0.9, 0.999),
    capturable: bool = False,
) -> torch.optim.Optimizer | MuonAdamW:
    """Reference SuperMimic split: hidden matrices use Muon at weight decay 0.1.

    Boundary matrices marked ``_non_muon``, biases and normalization parameters
    use Adam (muon_adam) or AdamW (muon). Plain adam and adamw are also available.
    """
    if kind not in ('adam', 'adamw', 'muon', 'muon_adam'):
        raise ValueError(f'Unknown optimizer {kind!r}')
    parameters = [p for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('Optimizer requires trainable parameters')
    matrices, other = [], []
    for parameter in parameters:
        target = matrices if kind.startswith('muon') and parameter.ndim == 2 and not getattr(parameter, '_non_muon', False) else other
        target.append(parameter)
    muon = torch.optim.Muon(matrices, lr=lr, weight_decay=0.1,
                            adjust_lr_fn='match_rms_adamw') if matrices else None
    if not other:
        return muon
    adam_type = torch.optim.Adam if kind in ('adam', 'muon_adam') else torch.optim.AdamW
    adam = adam_type(other, lr=lr, betas=adam_betas,
                     weight_decay=0.0 if adam_type is torch.optim.Adam else 0.01,
                     fused=all(p.is_cuda for p in other), capturable=capturable)
    if not matrices:
        return adam
    return MuonAdamW(adam, muon)
