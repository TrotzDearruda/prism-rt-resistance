"""Programme penalty.

Ref: Methods Sec. 6.1 Eq. (1), last summand -- L_prog(h) penalises the decoder for sparsity
and signs. Ref: Methods Sec. 6.6 -- sparsity is imposed by a group penalty over
metabolically coherent blocks rather than element-wise, because an element-wise penalty
selects isolated metabolites instead of a programme. The sign half of the penalty is what
makes the emission signed rather than merely sparse: a coefficient pointing the wrong way
is charged, so the direction of movement is part of what the decoder is fitted to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from prism.cohort.lexicon import Lexicon
from prism.errors import ConfigError


def group_norms(
    coefficients: Tensor, blocks: Sequence[Sequence[int]], weights: Sequence[float]
) -> Tensor:
    """Weighted L2 norm of each block, one column per block."""
    validate_blocks(blocks, weights, int(coefficients.shape[1]))
    columns: list[Tensor] = []
    for block, weight in zip(blocks, weights, strict=True):
        segment: Tensor = coefficients[:, list(block)]
        columns.append(weight * torch.linalg.vector_norm(segment, ord=2, dim=1))
    return torch.stack(columns, dim=1)


def validate_blocks(blocks: Sequence[Sequence[int]], weights: Sequence[float], width: int) -> None:
    if len(blocks) != len(weights):
        raise ConfigError("every block needs a group weight")
    flat = [index for block in blocks for index in block]
    if sorted(flat) != list(range(width)):
        raise ConfigError("the blocks must partition the coefficient width")
    if any(weight <= 0.0 for weight in weights):
        raise ConfigError("group weights must be positive")


@dataclass(frozen=True)
class GroupPenalty:
    """Group lasso over metabolically coherent blocks, optionally on the block norms only."""

    group_weight: float = 1.0
    square: bool = True
    adaptive: bool = True

    def __call__(self, coefficients: Tensor, lexicon: Lexicon) -> Tensor:
        if coefficients.dim() != 2 or int(coefficients.shape[1]) != lexicon.size:
            raise ConfigError("decoder coefficients must be (rows, vocabulary_size)")
        weights = (
            lexicon.block_weights() if self.adaptive else tuple(1.0 for _ in lexicon.index_blocks())
        )
        norms = group_norms(coefficients, lexicon.index_blocks(), weights)
        penalty: Tensor = (
            (norms * norms).sum(dim=1).mean() if self.square else norms.sum(dim=1).mean()
        )
        return self.group_weight * penalty


@dataclass(frozen=True)
class SignPenalty:
    """Charge a coefficient that points against the reported direction of movement."""

    sign_weight: float = 1.0

    def __call__(self, coefficients: Tensor, expected_signs: Sequence[int]) -> Tensor:
        if int(coefficients.shape[1]) != len(expected_signs):
            raise ConfigError("one expected sign is needed per vocabulary entry")
        direction: Tensor = torch.tensor(
            list(expected_signs), dtype=coefficients.dtype, device=coefficients.device
        )
        wrong: Tensor = torch.clamp(-direction.unsqueeze(0) * coefficients, min=0.0)
        return self.sign_weight * wrong.sum(dim=1).mean()


@dataclass(frozen=True)
class ProgrammePenalty:
    """L_prog(h) with the two halves kept visible for the ablation table."""

    group_weight: float = 1.0
    sign_weight: float = 0.05
    square: bool = True

    def __call__(
        self, coefficients: Tensor, lexicon: Lexicon, expected_signs: Sequence[int]
    ) -> Mapping[str, Tensor]:
        group = GroupPenalty(self.group_weight, self.square)(coefficients, lexicon)
        sign = SignPenalty(self.sign_weight)(coefficients, expected_signs)
        return {"programme": group + sign, "group": group, "sign": sign}


def sparsity_fraction(coefficients: Tensor, epsilon: float = 1e-6) -> float:
    """Share of coefficients that are effectively zero across a batch."""
    if coefficients.numel() == 0:
        raise ConfigError("a sparsity fraction needs a non-empty coefficient block")
    zero_rows: Tensor = (coefficients.abs() <= epsilon).to(torch.float32)
    return float(zero_rows.mean())


def block_activity(coefficients: Tensor, lexicon: Lexicon) -> Mapping[str, float]:
    """Mean block norm per programme, the quantity the coherence score is built on."""
    if coefficients.dim() != 2 or int(coefficients.shape[1]) != lexicon.size:
        raise ConfigError("decoder coefficients must be (rows, vocabulary_size)")
    activity: dict[str, float] = {}
    for position, block in enumerate(lexicon.index_blocks()):
        segment: Tensor = coefficients[:, list(block)]
        norms: Tensor = torch.linalg.vector_norm(segment, ord=2, dim=1)
        activity[lexicon.programme_ids[position]] = float(norms.mean())
    return activity


def effective_programmes(
    coefficients: Tensor, lexicon: Lexicon, threshold: float
) -> tuple[str, ...]:
    """Programmes whose mean block norm clears a threshold."""
    activity = block_activity(coefficients, lexicon)
    return tuple(name for name, value in activity.items() if value > threshold)
