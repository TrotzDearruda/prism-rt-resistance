"""Numerical utilities: seeding, patient-level resampling, second-moment algebra."""

from __future__ import annotations

from prism.numerics.boot import (
    BootstrapResult,
    bootstrap_functional,
    model_ensemble_summary,
    patient_blocks,
)
from prism.numerics.interval import Interval
from prism.numerics.linalg import (
    between_group_covariance,
    between_group_trace,
    centred,
    correlation,
    covariance,
    frobenius,
    gram,
    subspace_overlap,
    whitening_operator,
)
from prism.numerics.operating_point import (
    RuleInProfile,
    operating_point,
    prevalence_sensitivity,
    rule_in_at_deployment,
    rule_in_profile,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from prism.numerics.seed import (
    DEFAULT_SEED,
    SEED_ENSEMBLE,
    SeedState,
    ensemble_seeds,
    set_seed,
)

__all__ = [
    "DEFAULT_SEED",
    "Interval",
    "SEED_ENSEMBLE",
    "BootstrapResult",
    "RuleInProfile",
    "SeedState",
    "between_group_covariance",
    "between_group_trace",
    "bootstrap_functional",
    "centred",
    "correlation",
    "covariance",
    "ensemble_seeds",
    "frobenius",
    "gram",
    "model_ensemble_summary",
    "operating_point",
    "patient_blocks",
    "prevalence_sensitivity",
    "rule_in_at_deployment",
    "rule_in_profile",
    "sensitivity_at_specificity",
    "set_seed",
    "specificity_at_sensitivity",
    "subspace_overlap",
    "whitening_operator",
]
