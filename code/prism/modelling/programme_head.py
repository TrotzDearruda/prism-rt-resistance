"""Programme decoder.

Ref: Methods Sec. 6.6 -- h converts the representation into a sparse signed vector over
the metabolite vocabulary; the supported set is the programme pi, and the sign of each
coefficient states the direction in which that entry is predicted to move with respect to
radiosensitivity. Sparsity is imposed by a group penalty over metabolically coherent
blocks rather than element-wise, so the head emits a coefficient per vocabulary entry and
the block structure is applied in the objective rather than in the head.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.cohort.lexicon import Lexicon
from prism.errors import ConfigError
from prism.notation import ProgrammeEmission


@dataclass(frozen=True)
class SupportThreshold:
    """Rule that turns continuous coefficients into a supported set.

    The study fixes the emission as a subset with signs but does not print the cut, so the
    default combines a magnitude floor with a per-block retain count. Both are engineering
    defaults and both are recorded in the run header.
    """

    magnitude: float = 0.15
    per_block_min: int = 1
    per_block_max: int = 4

    def validate(self) -> None:
        if self.magnitude < 0.0:
            raise ConfigError("a support threshold cannot be negative")
        if not 0 <= self.per_block_min <= self.per_block_max:
            raise ConfigError("the per-block retain counts are inconsistent")


class ProgrammeDecoder(nn.Module):
    """h: fused representation to signed metabolite coefficients."""

    net: nn.Sequential
    out: nn.Linear

    def __init__(
        self,
        in_dim: int,
        vocabulary_size: int,
        hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim < 1 or vocabulary_size < 1 or hidden < 1:
            raise ConfigError("a programme decoder needs positive widths")
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, vocabulary_size)
        with torch.no_grad():
            self.out.weight.mul_(0.25)
            self.out.bias.zero_()
        self.vocabulary_size = vocabulary_size

    def forward(self, representation: Tensor) -> Tensor:
        hidden: Tensor = self.net(representation)
        coefficients: Tensor = self.out(hidden)
        return coefficients


def select_support(
    coefficients: Tensor,
    lexicon: Lexicon,
    threshold: SupportThreshold | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Pick the supported entries of each row, block by block."""
    resolved = threshold if threshold is not None else SupportThreshold()
    resolved.validate()
    if coefficients.dim() != 2 or coefficients.shape[1] != lexicon.size:
        raise ConfigError("decoder coefficients must be (rows, vocabulary_size)")
    magnitudes: Tensor = coefficients.detach().abs()
    selected: list[tuple[int, ...]] = []
    for row in range(magnitudes.shape[0]):
        chosen: list[int] = []
        for block in lexicon.index_blocks():
            inside = magnitudes[row, list(block)]
            order = torch.argsort(inside, descending=True)
            block_choices: list[int] = [
                int(block[int(index)])
                for index in order[: resolved.per_block_max]
                if float(inside[int(index)]) >= resolved.magnitude
            ]
            if len(block_choices) < resolved.per_block_min:
                best = int(order[0])
                block_choices = [int(block[best])]
            chosen.extend(block_choices)
        selected.append(tuple(sorted(chosen)))
    return tuple(selected)


def signs_for_support(
    coefficients: Tensor, support: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """Sign of each supported entry, snapped to plus or minus one."""
    signs: list[tuple[int, ...]] = []
    magnitudes = coefficients.detach()
    for row, indices in enumerate(support):
        row_signs = tuple(1 if float(magnitudes[row, index]) >= 0.0 else -1 for index in indices)
        signs.append(row_signs)
    return tuple(signs)


def emissions_from_coefficients(
    coefficients: Tensor,
    lexicon: Lexicon,
    record_ids: tuple[str, ...],
    threshold: SupportThreshold | None = None,
) -> tuple[ProgrammeEmission, ...]:
    """Turn decoder coefficients into the emitted programmes of a batch."""
    if len(record_ids) != int(coefficients.shape[0]):
        raise ConfigError("record identifiers must match the coefficient rows")
    support = select_support(coefficients, lexicon, threshold)
    signs = signs_for_support(coefficients, support)
    names = lexicon.names()
    emissions: list[ProgrammeEmission] = []
    for row, indices in enumerate(support):
        supported_names = tuple(names[index] for index in indices)
        sign_map = {names[index]: signs[row][position] for position, index in enumerate(indices)}
        programmes = tuple(sorted({lexicon.programme_of(index) for index in indices}))
        emissions.append(
            ProgrammeEmission(
                record_id=record_ids[row],
                supported=supported_names,
                signs=sign_map,
                programme_support=programmes,
                beta={},
                interval={},
            )
        )
    return tuple(emissions)


def block_norms(coefficients: Tensor, lexicon: Lexicon) -> Tensor:
    """L2 norm of each block, the input to the group penalty and to the coherence score."""
    if coefficients.dim() != 2 or coefficients.shape[1] != lexicon.size:
        raise ConfigError("decoder coefficients must be (rows, vocabulary_size)")
    columns: list[Tensor] = []
    for block in lexicon.index_blocks():
        segment: Tensor = coefficients[:, list(block)]
        columns.append(torch.linalg.vector_norm(segment, ord=2, dim=1))
    return torch.stack(columns, dim=1)


def support_sizes(support: tuple[tuple[int, ...], ...], lexicon: Lexicon) -> Mapping[str, float]:
    """Mean number of supported entries overall and per programme."""
    if not support:
        return {"overall": 0.0}
    overall = sum(len(row) for row in support) / float(len(support))
    per_programme: dict[str, float] = {"overall": overall}
    for position, block in enumerate(lexicon.index_blocks()):
        members = set(block)
        counts = [sum(1 for index in row if index in members) for row in support]
        per_programme[lexicon.programme_ids[position]] = sum(counts) / float(len(counts))
    return per_programme


def sign_agreement(
    coefficients: Tensor, lexicon: Lexicon, expected_signs: Mapping[str, int]
) -> float:
    """Share of supported entries whose sign matches the expected direction of movement."""
    support = select_support(coefficients, lexicon)
    signs = signs_for_support(coefficients, support)
    names = lexicon.names()
    agree = 0
    total = 0
    for row, indices in enumerate(support):
        for position, index in enumerate(indices):
            expected = expected_signs.get(names[index])
            if expected is None:
                continue
            total += 1
            if signs[row][position] == expected:
                agree += 1
    if total == 0:
        return 0.0
    return agree / float(total)
