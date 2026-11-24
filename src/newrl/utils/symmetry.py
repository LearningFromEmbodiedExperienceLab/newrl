"""Signed permutations for left–right reflection of flattened features."""

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class SymmetryTransform(nn.Module):
    def __init__(self, permutation: Sequence[int], signs: Sequence[int]) -> None:
        super().__init__()
        width = len(permutation)
        if sorted(permutation) != list(range(width)):
            raise ValueError("Symmetry indices must be a permutation")
        if len(signs) != width or any(sign not in (-1, 1) for sign in signs):
            raise ValueError("Each symmetry index must have a sign of -1 or 1")
        if any(
            permutation[partner] != index or signs[index] != signs[partner]
            for index, partner in enumerate(permutation)
        ):
            raise ValueError("Applying the reflection twice must recover the input")
        self.register_buffer(
            "indices", torch.tensor(permutation, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "signs", torch.tensor(signs, dtype=torch.int8), persistent=False
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.permute(values) * self.signs

    def permute(self, values: torch.Tensor) -> torch.Tensor:
        """Reflect features such as Gaussian scales that have no sign change."""
        if values.shape[-1] != self.indices.numel():
            raise ValueError(
                f"Expected {self.indices.numel()} symmetry features, got {values.shape[-1]}"
            )
        return values.index_select(-1, self.indices)

    @classmethod
    def cat(cls, transforms: Sequence["SymmetryTransform"]) -> "SymmetryTransform":
        indices = []
        signs = []
        for transform in transforms:
            offset = len(indices)
            indices.extend(index + offset for index in transform.indices.tolist())
            signs.extend(transform.signs.tolist())
        return cls(indices, signs)

    @classmethod
    def from_names(
        cls,
        labels: Sequence[str],
        partners: Mapping[str, tuple[str, int]],
        component_signs: tuple[int, ...] = (1,),
    ) -> "SymmetryTransform":
        """Bind name pairs to scene order and expand row-major vector features."""
        names = [label.rsplit("/", 1)[-1] for label in labels]
        positions = {name: index for index, name in enumerate(names)}
        if len(positions) != len(names):
            raise ValueError(f"Ambiguous symmetry labels in {labels}")
        width = len(component_signs)
        indices = []
        signs = []
        for name in names:
            if name not in partners:
                raise ValueError(f"No symmetry partner declared for {name!r}")
            partner, sign = partners[name]
            if partner not in positions:
                raise ValueError(
                    f"Symmetry partner {partner!r} for {name!r} was not selected"
                )
            offset = width * positions[partner]
            indices.extend(range(offset, offset + width))
            signs.extend(sign * component_sign for component_sign in component_signs)
        return cls(indices, signs)
