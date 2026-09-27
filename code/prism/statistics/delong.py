"""Paired comparison of two areas under the curve.

Ref: Methods Sec. 6.10 -- differences in discrimination are assessed with the paired DeLong
test over the pre-specified comparison set, with the multiple-testing burden handled by the
Holm correction of ``statistics.holm``. The implementation below is the structural-component
form: the placement values of each model are formed once, their covariance is read off the
two models together, and the paired statistic follows from that covariance rather than from a
difference of independent variances, which is what "paired" is doing in the name of the test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from prism.errors import ConfigError


def midrank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, the placement used by the fast form of the test."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    position = 0
    while position < array.size:
        end = position
        while end + 1 < array.size and sorted_values[end + 1] == sorted_values[position]:
            end += 1
        ranks[order[position : end + 1]] = 0.5 * (position + end) + 1.0
        position = end + 1
    return ranks


def structural_components(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two placement vectors the test is built from."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    positive = truth > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ConfigError("a paired comparison needs both classes present")
    ranks = midrank(values)
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    positive_component = (ranks[positive] - np.arange(1, n_positive + 1)) / float(n_negative)
    negative_component = 1.0 - (ranks[negative] - np.arange(1, n_negative + 1)) / float(n_positive)
    return positive_component, negative_component


def auc_from_components(positive_component: np.ndarray, negative_component: np.ndarray) -> float:
    return 0.5 * (float(positive_component.mean()) + float(negative_component.mean()))


def covariance_from_components(
    positive_a: np.ndarray,
    negative_a: np.ndarray,
    positive_b: np.ndarray,
    negative_b: np.ndarray,
) -> np.ndarray:
    """2x2 covariance of two area estimates on the same cases."""
    n_positive = positive_a.size
    n_negative = negative_a.size
    stacked_positive = np.stack([positive_a, positive_b], axis=1)
    stacked_negative = np.stack([negative_a, negative_b], axis=1)
    cov_positive = np.cov(stacked_positive, rowvar=False, ddof=1) / float(n_positive)
    cov_negative = np.cov(stacked_negative, rowvar=False, ddof=1) / float(n_negative)
    covariance: np.ndarray = np.asarray(cov_positive + cov_negative, dtype=np.float64)
    if covariance.shape != (2, 2):
        raise ConfigError("a paired covariance must be two by two")
    return covariance


@dataclass(frozen=True)
class PairedComparison:
    """One paired contrast, reported the way the family table reads it."""

    label: str
    auc_a: float
    auc_b: float
    difference: float
    standard_error: float
    z_statistic: float
    p_value: float
    n_rows: int
    n_positive: int

    def as_dict(self) -> dict[str, float | str]:
        return {
            "contrast": self.label,
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "z": self.z_statistic,
            "p_value": self.p_value,
            "n_rows": float(self.n_rows),
            "n_positive": float(self.n_positive),
        }

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        quantile = float(stats.norm.ppf(1.0 - 0.5 * (1.0 - level)))
        return (
            self.difference - quantile * self.standard_error,
            self.difference + quantile * self.standard_error,
        )


def paired_delong(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    *,
    label: str = "a-vs-b",
) -> PairedComparison:
    """Paired DeLong contrast between two score vectors on the same cases."""
    first = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    second = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    if not (first.size == second.size == truth.size):
        raise ConfigError("a paired contrast needs both score vectors on the same cases")
    positive_a, negative_a = structural_components(first, truth)
    positive_b, negative_b = structural_components(second, truth)
    covariance = covariance_from_components(positive_a, negative_a, positive_b, negative_b)
    auc_a = auc_from_components(positive_a, negative_a)
    auc_b = auc_from_components(positive_b, negative_b)
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    standard_error = float(np.sqrt(max(variance, 0.0)))
    difference = auc_a - auc_b
    if standard_error <= 0.0:
        z = 0.0 if difference == 0.0 else float(np.sign(difference)) * float("inf")
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z = difference / standard_error
        p_value = 2.0 * float(stats.norm.sf(abs(z)))
    n_positive = int((truth > 0.5).sum())
    return PairedComparison(
        label=label,
        auc_a=auc_a,
        auc_b=auc_b,
        difference=difference,
        standard_error=standard_error,
        z_statistic=z,
        p_value=p_value,
        n_rows=int(truth.size),
        n_positive=n_positive,
    )


def family_table(comparisons: Sequence[PairedComparison]) -> tuple[PairedComparison, ...]:
    if not comparisons:
        raise ConfigError("a comparison family needs at least one contrast")
    labels = [comparison.label for comparison in comparisons]
    if len(set(labels)) != len(labels):
        raise ConfigError("contrast labels must be distinct inside a family")
    return tuple(comparisons)


def pooled_auc_variance(scores: np.ndarray, labels: np.ndarray) -> float:
    """Variance of one area estimate, for a contrast that cannot be paired."""
    positive, negative = structural_components(scores, labels)
    n_positive = positive.size
    n_negative = negative.size
    var_positive = float(np.var(positive, ddof=1)) / float(n_positive)
    var_negative = float(np.var(negative, ddof=1)) / float(n_negative)
    return var_positive + var_negative


def unpaired_contrast(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    label: str = "unpaired",
) -> PairedComparison:
    """Two-sample contrast, used only where the partitions do not share cases."""
    positive_a, negative_a = structural_components(scores_a, labels_a)
    positive_b, negative_b = structural_components(scores_b, labels_b)
    auc_a = auc_from_components(positive_a, negative_a)
    auc_b = auc_from_components(positive_b, negative_b)
    variance = pooled_auc_variance(scores_a, labels_a) + pooled_auc_variance(scores_b, labels_b)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    difference = auc_a - auc_b
    if standard_error <= 0.0:
        z = 0.0 if difference == 0.0 else float(np.sign(difference)) * float("inf")
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z = difference / standard_error
        p_value = 2.0 * float(stats.norm.sf(abs(z)))
    return PairedComparison(
        label=label,
        auc_a=auc_a,
        auc_b=auc_b,
        difference=difference,
        standard_error=standard_error,
        z_statistic=z,
        p_value=p_value,
        n_rows=int(np.asarray(labels_a).size + np.asarray(labels_b).size),
        n_positive=int((np.asarray(labels_a) > 0.5).sum() + (np.asarray(labels_b) > 0.5).sum()),
    )
