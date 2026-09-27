"""Pre-specified analysis plan.

Ref: Methods Sec. 6.8 and Sec. 6.9 -- three comparisons are recognised beforehand, the success
criterion is conjunctive, two points of reference are mandatory, and the multiplicity family, the
null band and the equivalence margin are fixed before any held-out partition is inspected. This
module assembles those declarations into one object so that a run can be checked against the plan
it claims to follow instead of against a plan chosen afterwards.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prism.errors import ProtocolError
from prism.objectives.feasibility import FeasibilityFloor
from prism.statistics.equivalence import SuccessCriterion
from prism.statistics.family import (
    ContrastSpec,
    conditional_value_reading,
    family,
    ordered_p_values,
    synergy_statistic,
    validate_family,
)
from prism.statistics.holm import family_size_matches, holm_step_down
from prism.statistics.nullband import NullBand


@dataclass(frozen=True)
class AnalysisPlan:
    """Everything the study fixed before fitting, in one place."""

    margin: float
    null_band: NullBand
    discrimination_floor: FeasibilityFloor
    family: tuple[ContrastSpec, ...]
    interval_nominal_coverage: float
    references: tuple[str, ...]

    def validate(self) -> None:
        if self.margin <= 0.0:
            raise ProtocolError("the equivalence margin must be positive")
        self.null_band.validate()
        if not 0.0 < self.interval_nominal_coverage < 1.0:
            raise ProtocolError("the interval coverage target must lie strictly inside (0, 1)")
        validate_family(self.family)
        if not family_size_matches([spec.name for spec in self.family]):
            raise ProtocolError(
                "the analysis plan fixes five contrasts; this plan carries " f"{len(self.family)}"
            )
        if len(self.references) < 2:
            raise ProtocolError(
                "the study names two mandatory points of reference: the standard of care and an "
                "architectural comparison"
            )

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "equivalence_margin": self.margin,
            "null_band": self.null_band.value,
            "discrimination_floor": self.discrimination_floor.d_min,
            "interval_nominal_coverage": self.interval_nominal_coverage,
            "family": [spec.as_dict() for spec in self.family],
            "references": list(self.references),
        }


def default_plan(
    margin: float,
    null_band: float,
    discrimination_floor: float,
    interval_nominal_coverage: float,
    references: Sequence[str] = ("clinical-standard-of-care", "convolutional-benchmark"),
) -> AnalysisPlan:
    plan = AnalysisPlan(
        margin=float(margin),
        null_band=NullBand(value=float(null_band)),
        discrimination_floor=FeasibilityFloor(d_min=float(discrimination_floor)),
        family=family(),
        interval_nominal_coverage=float(interval_nominal_coverage),
        references=tuple(references),
    )
    plan.validate()
    return plan


def criterion_from_measurements(
    primary_constrained: float,
    primary_unconstrained: float,
    external_constrained: float,
    external_unconstrained: float,
    margin: float,
) -> SuccessCriterion:
    """Assemble the conjunctive criterion from the four measured values.

    Ref: Methods Sec. 6.9 -- the primary limb is an equivalence claim about the primary partition
    and the second limb is a superiority claim on the external partition; the criterion is
    satisfied only when both hold.
    """
    return SuccessCriterion(
        primary_difference=float(primary_constrained) - float(primary_unconstrained),
        margin=float(margin),
        external_difference=float(external_constrained) - float(external_unconstrained),
    )


@dataclass(frozen=True)
class FamilyOutcome:
    """The family after the Holm correction, next to the criterion and the conditional reading."""

    adjusted_p: tuple[Mapping[str, object], ...]
    criterion: Mapping[str, object]
    conditional: Mapping[str, float]
    synergy: Mapping[str, float]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "family": list(self.adjusted_p),
            "criterion": dict(self.criterion),
            "conditional": dict(self.conditional),
            "synergy": dict(self.synergy),
        }


def adjudicate_plan(
    plan: AnalysisPlan,
    raw_p: Mapping[str, float],
    primary_constrained: float,
    primary_unconstrained: float,
    external_constrained: float,
    external_unconstrained: float,
    alignment_cost: float,
    invariance_cost: float,
    both_cost: float,
    conditional_gain: Mapping[str, float],
    retained: Sequence[str],
) -> FamilyOutcome:
    """Run the pre-specified readings in one call, in the order the study reports them."""
    plan.validate()
    values = ordered_p_values(raw_p)
    step_down = holm_step_down(values, [spec.name for spec in plan.family])
    criterion = criterion_from_measurements(
        primary_constrained,
        primary_unconstrained,
        external_constrained,
        external_unconstrained,
        plan.margin,
    )
    return FamilyOutcome(
        adjusted_p=step_down.as_table(),
        criterion=criterion.as_dict(),
        conditional=dict(
            conditional_value_reading(conditional_gain, plan.null_band.value, retained)
        ),
        synergy=dict(synergy_statistic(alignment_cost, invariance_cost, both_cost)),
    )


def plan_digest(plan: AnalysisPlan) -> str:
    """Stable digest of the plan, written into the run header before fitting starts."""
    payload = json.dumps(plan.as_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def report_references(plan: AnalysisPlan) -> Mapping[str, str]:
    """What each mandatory reference contributes to the comparison.

    Ref: Methods Sec. 6.9 -- the standard of care reference decides whether the model can be used
    at all, and the architectural reference keeps the comparison inside a family of encoders.
    """
    plan.validate()
    return {
        "clinical-standard-of-care": (
            "the operating standard against which clinical usefulness is read"
        ),
        "convolutional-benchmark": (
            "a second architecture family, so the comparison is not confined to vision transformers"
        ),
    }
