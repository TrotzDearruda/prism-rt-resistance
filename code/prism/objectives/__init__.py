"""Objective terms of the constrained problem."""

from __future__ import annotations

from prism.objectives.alignment import (
    AlignmentLoss,
    covariance_matrix,
    gram_matrix,
    regional_alignment,
)
from prism.objectives.feasibility import (
    FeasibilityFloor,
    FeasibilityLedger,
    FeasibilityVerdict,
    assert_feasible_at_end,
    evaluate_epoch,
    restore_snapshot,
    state_snapshot,
    trade_off_point,
)
from prism.objectives.invariance import (
    AdversarialPenalty,
    AttributionSummary,
    VariancePenalty,
    between_group_covariance,
    flatten_attribution,
    penalty_for_mode,
)
from prism.objectives.outcome import (
    OutcomeLoss,
    balanced_positive_weight,
    brier_decomposition_free,
    implied_prevalence,
    outcome_accuracy,
)
from prism.objectives.programme import (
    GroupPenalty,
    ProgrammePenalty,
    SignPenalty,
    block_activity,
    effective_programmes,
    group_norms,
    sparsity_fraction,
    validate_blocks,
)
from prism.objectives.total import (
    ConstrainedObjective,
    ObjectiveResult,
    ObjectiveWeights,
    blend,
    effective_floor,
    zero_terms,
)

__all__ = [
    "AdversarialPenalty",
    "AlignmentLoss",
    "AttributionSummary",
    "ConstrainedObjective",
    "FeasibilityFloor",
    "FeasibilityLedger",
    "FeasibilityVerdict",
    "GroupPenalty",
    "ObjectiveResult",
    "ObjectiveWeights",
    "OutcomeLoss",
    "ProgrammePenalty",
    "SignPenalty",
    "VariancePenalty",
    "assert_feasible_at_end",
    "balanced_positive_weight",
    "between_group_covariance",
    "blend",
    "block_activity",
    "brier_decomposition_free",
    "covariance_matrix",
    "effective_floor",
    "effective_programmes",
    "evaluate_epoch",
    "flatten_attribution",
    "gram_matrix",
    "group_norms",
    "implied_prevalence",
    "outcome_accuracy",
    "penalty_for_mode",
    "regional_alignment",
    "restore_snapshot",
    "sparsity_fraction",
    "state_snapshot",
    "trade_off_point",
    "validate_blocks",
    "zero_terms",
]
