"""Operating-point arithmetic and rule-in values.

Ref: Results Sec. 2.1 -- the response rate fixes the prevalence at which a test is read, and at
that prevalence the positive predictive value of a test cannot be near one, so rule-in
statements are reported as positive predictive values rather than as sensitivities. The
arithmetic that turns a sensitivity, a specificity and a prevalence into those values is
shared by the prevalence section and by the reader study, so it lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError


@dataclass(frozen=True)
class RuleInProfile:
    """Sensitivity, specificity and prevalence combined into the numbers a pathway uses."""

    prevalence: float
    sensitivity: float
    specificity: float

    def ppv(self) -> float:
        hit = self.sensitivity * self.prevalence
        miss = (1.0 - self.specificity) * (1.0 - self.prevalence)
        denominator = hit + miss
        if denominator <= 0.0:
            raise ConfigError("a positive predictive value is undefined at this operating point")
        return hit / denominator

    def npv(self) -> float:
        correct_rejection = self.specificity * (1.0 - self.prevalence)
        false_rejection = (1.0 - self.sensitivity) * self.prevalence
        denominator = correct_rejection + false_rejection
        if denominator <= 0.0:
            raise ConfigError("a negative predictive value is undefined at this operating point")
        return correct_rejection / denominator

    def odds_ratio(self) -> float:
        numerator = self.sensitivity * self.specificity
        denominator = (1.0 - self.sensitivity) * (1.0 - self.specificity)
        if denominator <= 0.0:
            return float("inf")
        return numerator / denominator

    def as_dict(self) -> dict[str, float]:
        return {
            "prevalence": self.prevalence,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "positive_predictive_value": self.ppv(),
            "negative_predictive_value": self.npv(),
        }


def _split(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    if values.size != truth.size or values.size == 0:
        raise ConfigError("an operating point needs aligned scores and labels")
    positive = truth > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ConfigError("an operating point needs both classes present")
    return values[positive], values[negative]


def sensitivity_at_specificity(
    scores: np.ndarray, labels: np.ndarray, target_specificity: float
) -> float:
    """Sensitivity at a specificity the pathway can actually operate at.

    The cut is the quantile of the negative scores at the target specificity, because the
    specificity is the share of negatives that fall below the cut. Taking the complementary quantile
    instead returns the operating point with the opposite specificity, which is the quantity the
    rule-in arithmetic of Results Sec. 2.1 is most sensitive to.
    """
    if not 0.0 <= target_specificity <= 1.0:
        raise ConfigError("a target specificity must lie in [0, 1]")
    positive, negative = _split(scores, labels)
    threshold = float(np.quantile(negative, target_specificity))
    return float((positive >= threshold).mean())


def specificity_at_sensitivity(
    scores: np.ndarray, labels: np.ndarray, target_sensitivity: float
) -> float:
    if not 0.0 <= target_sensitivity <= 1.0:
        raise ConfigError("a target sensitivity must lie in [0, 1]")
    positive, negative = _split(scores, labels)
    threshold = float(np.quantile(positive, 1.0 - target_sensitivity))
    return float((negative < threshold).mean())


def operating_point(
    scores: np.ndarray, labels: np.ndarray, target_specificity: float
) -> tuple[float, float]:
    """Sensitivity and the specificity actually achieved at that cut."""
    sensitivity = sensitivity_at_specificity(scores, labels, target_specificity)
    specificity = specificity_at_sensitivity(scores, labels, sensitivity)
    return sensitivity, specificity


def rule_in_profile(
    scores: np.ndarray,
    labels: np.ndarray,
    prevalence: float,
    target_specificity: float = 1.0,
) -> RuleInProfile:
    """Rule-in profile at a prevalence, taken at the cut that holds the named specificity."""
    if not 0.0 < prevalence < 1.0:
        raise ConfigError("a prevalence must lie strictly inside (0, 1)")
    sensitivity, specificity = operating_point(scores, labels, target_specificity)
    return RuleInProfile(prevalence=prevalence, sensitivity=sensitivity, specificity=specificity)


def rule_in_at_deployment(
    scores: np.ndarray, labels: np.ndarray, prevalence: float
) -> RuleInProfile:
    """Rule-in profile at the deployment prevalence, at maximum specificity."""
    return rule_in_profile(scores, labels, prevalence, target_specificity=1.0)


def prevalence_sensitivity(profile: RuleInProfile, prevalence: float) -> RuleInProfile:
    """Re-read a fixed operating point at a different prevalence.

    Ref: Introduction, para 2 -- the low complete-response rate makes response testing sensitive
    to prevalence rather than to accuracy, and total neoadjuvant therapy has moved that
    prevalence over time, so the same sensitivity and specificity are read at more than one rate.
    """
    return RuleInProfile(
        prevalence=prevalence,
        sensitivity=profile.sensitivity,
        specificity=profile.specificity,
    )
