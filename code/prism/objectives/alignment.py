"""Spatial-metabolic alignment term.

Ref: Methods Sec. 6.4 -- the alignment term both draws the fused representation of a region
towards the metabolite abundance of the same area and pulls the second-moment structure of
the region representations towards the metabolite representation. The manuscript states why
both halves are needed: matching alone admits a representation in which metabolite
information sits on arbitrary axes, and it is the covariance match that gives the axes a
metabolic reading, which the sparse decoder of Sec. 6.6 then depends on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from prism.errors import ConfigError


def _normalise(rows: Tensor) -> Tensor:
    norm: Tensor = torch.linalg.vector_norm(rows, ord=2, dim=1, keepdim=True).clamp(min=1e-6)
    scaled: Tensor = rows / norm
    return scaled


def gram_matrix(rows: Tensor) -> Tensor:
    """Row Gram matrix normalised by the row count."""
    if rows.dim() != 2:
        raise ConfigError("a Gram matrix reads a two-dimensional row block")
    product: Tensor = rows @ rows.T
    scaled: Tensor = product / float(max(int(rows.shape[0]), 1))
    return scaled


def covariance_matrix(rows: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Column covariance of a row block with a ridge on the diagonal."""
    if rows.dim() != 2:
        raise ConfigError("a covariance reads a two-dimensional row block")
    centred: Tensor = rows - rows.mean(dim=0, keepdim=True)
    product: Tensor = centred.T @ centred
    cov: Tensor = product / float(max(int(rows.shape[0]) - 1, 1))
    ridge: Tensor = eps * torch.eye(int(cov.shape[0]), dtype=cov.dtype, device=cov.device)
    return cov + ridge


@dataclass(frozen=True)
class AlignmentLoss:
    """Matching plus second-moment match.

    Ref: Methods Sec. 6.4. ``covariance_weight`` is the knob the study does not print; the
    default keeps the two halves on comparable scales, and the weight is recorded in the
    run header so an ablation can move it without touching code. ``temperature`` sharpens the
    matching term by scaling a non-negative disagreement, so the term cannot go negative and a
    negative value always means the input was malformed rather than well matched.
    """

    temperature: float = 0.1
    covariance_weight: float = 1.0
    matching_weight: float = 1.0
    standardise: bool = True

    def matching(self, representation: Tensor, metabolite: Tensor) -> Tensor:
        """Row-wise agreement between a representation and its metabolite block."""
        _check_alignment_shapes(representation, metabolite)
        left = representation
        right = metabolite
        if self.standardise:
            left = _normalise(left - left.mean(dim=0, keepdim=True))
            right = _normalise(right - right.mean(dim=0, keepdim=True))
        else:
            left = _normalise(left)
            right = _normalise(right)
        similarity: Tensor = (left * right).sum(dim=1)
        disagreement: Tensor = 1.0 - similarity
        return disagreement.mean() / max(self.temperature, 1e-6)

    def second_moment(self, representation: Tensor, metabolite: Tensor) -> Tensor:
        """Distance between two covariance structures, averaged over both readings."""
        _check_pair_rows(representation, metabolite)
        left = covariance_matrix(representation)
        right = covariance_matrix(metabolite)
        if left.shape != right.shape:
            raise ConfigError(
                "the second-moment match needs representations and metabolites of one width"
            )
        forward: Tensor = torch.linalg.matrix_norm(left - right, ord="fro")
        reverse: Tensor = torch.linalg.matrix_norm(right - left, ord="fro")
        scale = torch.sqrt((right * right).sum()).clamp(min=1e-6)
        return 0.5 * (forward + reverse) / scale

    def __call__(self, representation: Tensor, metabolite: Tensor) -> Mapping[str, Tensor]:
        match = self.matching_weight * self.matching(representation, metabolite)
        moment = self.covariance_weight * self.second_moment(representation, metabolite)
        return {"alignment": match + moment, "matching": match, "second_moment": moment}


def _check_alignment_shapes(representation: Tensor, metabolite: Tensor) -> None:
    if representation.dim() != 2 or metabolite.dim() != 2:
        raise ConfigError("the alignment term reads two row blocks")
    if representation.shape[0] != metabolite.shape[0]:
        raise ConfigError("the alignment term needs one metabolite block per row")
    if representation.shape[1] != metabolite.shape[1]:
        raise ConfigError(
            "the alignment term matches a read-out against a metabolite block of the same width, "
            "so an operand with the encoder's width and one with the vocabulary's width cannot be "
            "paired"
        )


def _check_pair_rows(representation: Tensor, metabolite: Tensor) -> None:
    if representation.dim() != 2 or metabolite.dim() != 2:
        raise ConfigError("the second-moment match reads two row blocks")
    if int(representation.shape[0]) < 3 or int(metabolite.shape[0]) < 3:
        raise ConfigError("a second-moment match needs at least three rows on each side")


def regional_alignment(
    representation_by_region: Mapping[str, Tensor],
    metabolite_by_region: Mapping[str, Tensor],
    loss: AlignmentLoss,
) -> Mapping[str, Tensor]:
    """Average the alignment term over regions.

    Ref: Methods Sec. 6.4 -- the quantity being matched is a region's structure, so the
    per-region terms are averaged rather than pooled into one covariance over a mixed batch.
    """
    if set(representation_by_region) != set(metabolite_by_region):
        raise ConfigError("each region needs a representation block and a metabolite block")
    if not representation_by_region:
        raise ConfigError("no region was supplied to the alignment term")
    collected: dict[str, list[Tensor]] = {"alignment": [], "matching": [], "second_moment": []}
    for region in sorted(representation_by_region):
        terms = loss(representation_by_region[region], metabolite_by_region[region])
        for key, value in terms.items():
            collected[key].append(value)
    return {key: torch.stack(values).mean() for key, values in collected.items()}
