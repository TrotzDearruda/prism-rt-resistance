"""Cross-setting invariance penalty.

Ref: Methods Sec. 6.5 -- the penalty reduces the share of Z that the care setting can
explain without shrinking the part that carries the response. Two readings of the penalty
are implemented because the ablation of Table 2 carries both: a variance penalty that
shrinks the between-site covariance of the representation, and an adversarial penalty that
learns a site classifier while acting against the encoder through a reversed gradient.
Ref: Table 3 Panel B -- the site-attribution probe is the read-out that says whether the
penalty did anything, and ``AttributionSummary`` here is the container the harness fills.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from prism.errors import ConfigError


def between_group_covariance(rows: Tensor, group_index: Tensor, n_groups: int) -> Tensor:
    """Covariance of the group centroids of a row block."""
    if rows.dim() != 2:
        raise ConfigError("the between-group covariance reads a two-dimensional block")
    if group_index.dim() != 1 or int(group_index.shape[0]) != int(rows.shape[0]):
        raise ConfigError("every row needs exactly one group index")
    if n_groups < 2:
        raise ConfigError("a between-group quantity needs at least two groups")
    width = int(rows.shape[1])
    centres = torch.zeros((n_groups, width), dtype=rows.dtype, device=rows.device)
    counts = torch.zeros(n_groups, dtype=rows.dtype, device=rows.device)
    centres.index_add_(0, group_index, rows)
    counts.index_add_(0, group_index, torch.ones_like(group_index, dtype=rows.dtype))
    present = counts > 0
    centres = torch.where(
        present.unsqueeze(1), centres / counts.clamp(min=1.0).unsqueeze(1), centres
    )
    active: Tensor = centres[present]
    if int(active.shape[0]) < 2:
        return torch.zeros((width, width), dtype=rows.dtype, device=rows.device)
    centred: Tensor = active - active.mean(dim=0, keepdim=True)
    product: Tensor = centred.T @ centred
    cov: Tensor = product / float(int(active.shape[0]) - 1)
    return cov


@dataclass(frozen=True)
class VariancePenalty:
    """Shrink the trace of the between-site covariance of the representation.

    Ref: Methods Sec. 6.5, first reading. The trace is normalised by the total variance of
    the block so that the penalty cannot be reduced by shrinking the representation overall,
    which would let the encoder satisfy the penalty while discarding the response as well.
    """

    normalise: bool = True

    def __call__(self, rows: Tensor, site_index: Tensor, n_sites: int) -> Mapping[str, Tensor]:
        between = between_group_covariance(rows, site_index, n_sites)
        trace_between: Tensor = torch.diagonal(between).sum()
        if not self.normalise:
            return {"invariance": trace_between, "between_trace": trace_between}
        centred: Tensor = rows - rows.mean(dim=0, keepdim=True)
        trace_total: Tensor = (centred * centred).sum() / float(max(int(rows.shape[0]) - 1, 1))
        ratio: Tensor = trace_between / trace_total.clamp(min=1e-6)
        return {"invariance": ratio, "between_trace": trace_between}


@dataclass(frozen=True)
class AdversarialPenalty:
    """Cross-entropy of the site classifier, with the encoder seeing the reversed gradient.

    Ref: Methods Sec. 6.5, second reading. The classifier is optimised towards the true
    settings while ``reverse_gradient`` sends the encoder the opposite direction, which is
    what makes the two halves of the objective conflict rather than cooperate.
    """

    label_smoothing: float = 0.0

    def __call__(self, site_logits: Tensor, site_index: Tensor) -> Mapping[str, Tensor]:
        if site_logits.dim() != 2:
            raise ConfigError("the site classifier emits one score per setting and row")
        if int(site_logits.shape[0]) != int(site_index.shape[0]):
            raise ConfigError("every row needs one site label")
        loss: Tensor = torch.nn.functional.cross_entropy(
            site_logits,
            site_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return {"invariance": loss, "adversarial": loss}


@dataclass(frozen=True)
class AttributionSummary:
    """Site and region attribution read off the same representations.

    Ref: Table 3 Panel B -- a representation that is well constrained should retain little
    site-attributable variance while keeping the outcome discrimination.
    """

    site_accuracy: float
    region_accuracy: float
    outcome_auc: float

    def site_excess_over_chance(self, n_sites: int) -> float:
        if n_sites < 2:
            raise ConfigError("chance level needs at least two settings")
        return self.site_accuracy - 1.0 / float(n_sites)

    def as_dict(self) -> Mapping[str, float]:
        return {
            "site_probe_accuracy": self.site_accuracy,
            "region_probe_accuracy": self.region_accuracy,
            "outcome_auc_retained": self.outcome_auc,
        }


def flatten_attribution(summary: AttributionSummary, n_sites: int) -> Mapping[str, float]:
    values = dict(summary.as_dict())
    values["site_excess_over_chance"] = summary.site_excess_over_chance(n_sites)
    return values


def penalty_for_mode(
    mode: str,
    rows: Tensor,
    site_index: Tensor,
    n_sites: int,
    site_logits: Tensor | None = None,
) -> Mapping[str, Tensor]:
    """Resolve the configured reading of the penalty."""
    if mode == "variance":
        return VariancePenalty()(rows, site_index, n_sites)
    if mode == "adversarial":
        if site_logits is None:
            raise ConfigError("the adversarial reading needs the site classifier output")
        return AdversarialPenalty()(site_logits, site_index)
    raise ConfigError(f"unknown invariance reading {mode!r}")
