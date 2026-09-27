"""Equivalence against a pre-declared margin.

Ref: Methods Sec. 6.9 -- the constrained configuration must sit within an equivalence range of
0.015 in the area under the curve of its equally capable unconstrained counterpart, and the
success criterion is conjunctive with the external-partition comparison. A parity claim is
therefore not a failure to reject a null of no difference; it is a test in the direction of
equivalence, which is what the two one-sided procedure below expresses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats

from prism.errors import ConfigError

DEFAULT_MARGIN = 0.015


def equivalence_interval(
    difference: float, standard_error: float, level: float = 0.95
) -> tuple[float, float]:
    if standard_error < 0.0:
        raise ConfigError("a standard error cannot be negative")
    quantile = float(stats.norm.ppf(1.0 - 0.5 * (1.0 - level)))
    return difference - quantile * standard_error, difference + quantile * standard_error


@dataclass(frozen=True)
class EquivalenceResult:
    """One two-one-sided-tests reading of a margin claim."""

    difference: float
    margin: float
    standard_error: float
    lower_test_p: float
    upper_test_p: float
    p_value: float
    equivalent: bool
    interval: tuple[float, float]

    def within_margin(self) -> bool:
        return abs(self.difference) <= self.margin

    def as_dict(self) -> Mapping[str, float | bool | tuple[float, float]]:
        return {
            "difference": self.difference,
            "margin": self.margin,
            "standard_error": self.standard_error,
            "lower_test_p": self.lower_test_p,
            "upper_test_p": self.upper_test_p,
            "p_value": self.p_value,
            "equivalent": self.equivalent,
            "interval": self.interval,
        }

    def margin_used(self) -> float:
        """Share of the margin consumed by the observed difference."""
        if self.margin <= 0.0:
            raise ConfigError("a margin must be positive to report consumption")
        return abs(self.difference) / self.margin


def two_one_sided(
    difference: float,
    standard_error: float,
    margin: float = DEFAULT_MARGIN,
    level: float = 0.95,
) -> EquivalenceResult:
    """Two one-sided tests against a symmetric margin."""
    if margin <= 0.0:
        raise ConfigError("an equivalence margin must be positive")
    if standard_error < 0.0:
        raise ConfigError("a standard error cannot be negative")
    if standard_error == 0.0:
        lower_p = 0.0 if difference > -margin else 1.0
        upper_p = 0.0 if difference < margin else 1.0
        p_value = max(lower_p, upper_p)
    else:
        lower_z = (difference + margin) / standard_error
        upper_z = (margin - difference) / standard_error
        lower_p = float(stats.norm.sf(lower_z))
        upper_p = float(stats.norm.sf(upper_z))
        p_value = max(lower_p, upper_p)
    alpha = 1.0 - level
    return EquivalenceResult(
        difference=difference,
        margin=margin,
        standard_error=standard_error,
        lower_test_p=lower_p,
        upper_test_p=upper_p,
        p_value=p_value,
        equivalent=bool(lower_p < alpha and upper_p < alpha),
        interval=equivalence_interval(difference, standard_error, level),
    )


@dataclass(frozen=True)
class SuccessCriterion:
    """The conjunctive pre-specified criterion.

    Ref: Methods Sec. 6.9 and Results Sec. 2.1 -- the criterion has two limbs and is satisfied
    only when both hold: the primary-partition discrimination of the constrained configuration
    is inside the margin, and the constrained configuration does better on the external
    partition. A run reports the two limbs separately, because a criterion reported only as
    satisfied or not cannot show which limb carried it.
    """

    primary_difference: float
    margin: float
    external_difference: float

    def primary_limb(self) -> bool:
        return self.primary_difference >= -self.margin

    def external_limb(self) -> bool:
        return self.external_difference > 0.0

    def satisfied(self) -> bool:
        return self.primary_limb() and self.external_limb()

    def margin_slack(self) -> float:
        """How much of the parity margin was left unused, on the primary limb."""
        return self.margin + self.primary_difference

    def as_dict(self) -> Mapping[str, float | bool]:
        return {
            "primary_difference": self.primary_difference,
            "margin": self.margin,
            "primary_limb": self.primary_limb(),
            "external_difference": self.external_difference,
            "external_limb": self.external_limb(),
            "satisfied": self.satisfied(),
            "margin_slack": self.margin_slack(),
        }


def compare_with_reference(external_difference: float, reference_difference: float) -> float:
    """Margin of the constrained model against the strongest named comparator.

    Ref: Results Sec. 4.1 -- the criterion also reports the external margin against the
    strongest standard, so a single difference is read against two references.
    """
    return external_difference - reference_difference


def reachability_check(margin: float, half_width: float) -> Mapping[str, float]:
    """Whether the margin is wide enough to be reached given the interval width.

    A margin narrower than the interval that estimates it cannot be claimed either way, so a
    run that reports a parity claim next to a wide interval is reporting a claim its own
    design cannot resolve.
    """
    if margin <= 0.0:
        raise ConfigError("a margin must be positive to check reachability")
    if half_width < 0.0:
        raise ConfigError("a half-width cannot be negative")
    return {
        "margin": margin,
        "interval_half_width": half_width,
        "margin_over_interval": margin / half_width if half_width > 0.0 else float("inf"),
        "resolvable": float(margin > half_width),
    }


def per_patient_difference(scores_a: np.ndarray, scores_b: np.ndarray, groups: np.ndarray) -> float:
    """Mean difference between two score vectors, averaged within a patient first.

    Ref: Methods Sec. 6.10 -- every partition is defined at the patient level, so a contrast
    that cannot be formed as a paired test is still averaged over patients rather than over
    records, and a patient contributing several records does not weigh more than one who
    contributes a single record.
    """
    first = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    second = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    labels = np.asarray(groups, dtype=object).reshape(-1)
    if first.shape != second.shape or first.size != labels.size:
        raise ConfigError("a per-patient difference needs aligned score vectors and patients")
    if first.size == 0:
        raise ConfigError("a per-patient difference needs at least one record")
    buckets: dict[str, list[float]] = {}
    for position in range(first.size):
        buckets.setdefault(str(labels[position]), []).append(
            float(first[position]) - float(second[position])
        )
    per_patient = [float(np.mean(values)) for values in buckets.values()]
    return float(np.mean(per_patient))


def per_patient_standard_error(
    scores_a: np.ndarray, scores_b: np.ndarray, groups: np.ndarray
) -> float:
    """Standard error of the per-patient difference, over patients."""
    first = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    second = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    labels = np.asarray(groups, dtype=object).reshape(-1)
    buckets: dict[str, list[float]] = {}
    for position in range(first.size):
        buckets.setdefault(str(labels[position]), []).append(
            float(first[position]) - float(second[position])
        )
    per_patient = np.asarray(
        [float(np.mean(values)) for values in buckets.values()], dtype=np.float64
    )
    if per_patient.size < 2:
        raise ConfigError("a per-patient standard error needs at least two patients")
    return float(per_patient.std(ddof=1) / float(np.sqrt(per_patient.size)))
