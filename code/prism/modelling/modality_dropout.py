"""Modality dropout.

Ref: Methods Sec. 6.3 and Algorithm 1 step 5 -- a dropout scheme varies the modalities used
during training so that the model can be estimated from any combination of streams a site
happens to have. A stream the site cannot acquire is absent by construction and is never
switched on; a stream the site does have is dropped with the configured probability,
subject to at least one stream surviving.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from prism.errors import ConfigError
from prism.notation import MODALITY_STREAMS


@dataclass(frozen=True)
class DropoutPlan:
    """Which stream is visible for which observation in one mini-batch."""

    present: Tensor
    order: tuple[str, ...]
    n_dropped: int
    n_rescued: int

    def visible(self, row: int) -> tuple[str, ...]:
        row_mask = self.present[row]
        return tuple(name for column, name in enumerate(self.order) if bool(row_mask[column]))


def draw_dropout_plan(
    availability: Tensor,
    probability: float,
    generator: torch.Generator | None = None,
    order: Sequence[str] = MODALITY_STREAMS,
) -> DropoutPlan:
    """Draw a dropout plan given a site's availability matrix.

    ``availability`` is (batch, n_streams) and records what the site could acquire.
    """
    if availability.dim() != 2:
        raise ConfigError("the availability tensor must be (batch, n_streams)")
    if availability.shape[1] != len(order):
        raise ConfigError("the availability tensor must cover every configured stream")
    if not 0.0 <= probability < 1.0:
        raise ConfigError("a dropout probability must lie in [0, 1)")

    source = availability.to(torch.bool)
    uniform = torch.rand(source.shape, generator=generator)
    keep = uniform >= probability
    present = source & keep

    rescued = 0
    survivors = present.sum(dim=1)
    empty = survivors == 0
    if bool(empty.any()) and bool(source.any(dim=1).all()):
        fallback = torch.multinomial(
            source[empty].to(torch.float32), num_samples=1, generator=generator
        )
        indices = torch.nonzero(empty, as_tuple=False).squeeze(-1)
        present = present.clone()
        present[indices, fallback.squeeze(-1)] = True
        rescued = int(indices.numel())

    dropped = int((source & ~present).sum().item())
    return DropoutPlan(present=present, order=tuple(order), n_dropped=dropped, n_rescued=rescued)


def availability_tensor(
    availability: Sequence[tuple[str, ...]],
    order: Sequence[str] = MODALITY_STREAMS,
) -> Tensor:
    """Turn per-observation modality lists into an indicator tensor."""
    rows: list[list[float]] = []
    for entry in availability:
        rows.append([1.0 if name in entry else 0.0 for name in order])
    if not rows:
        return torch.zeros((0, len(order)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def coverage_histogram(plan: DropoutPlan) -> Mapping[int, int]:
    """How many observations are left with each number of visible streams."""
    counts: dict[int, int] = {}
    totals = plan.present.sum(dim=1).to(torch.int64)
    for value in totals.tolist():
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def any_subset_estimable(plan: DropoutPlan) -> bool:
    """Whether every observation in the batch retains at least one modality."""
    return bool((plan.present.sum(dim=1) > 0).all())
