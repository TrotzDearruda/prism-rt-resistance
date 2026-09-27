"""Discrimination and ranking metrics.

Ref: Table 1 -- discrimination is reported as the area under the receiver-operating
characteristic curve, with area under the precision-recall curve carried beside it for the
annotated auxiliary series of Table 5. Ref: Methods Sec. 6.10 -- the same values feed the
paired DeLong test, so they are returned as per-patient scores rather than only as a
summary number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError


def _as_vectors(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    if values.size != truth.size:
        raise ConfigError("scores and labels must share a length")
    if values.size == 0:
        raise ConfigError("a discrimination value needs at least one row")
    return values, truth


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve from the rank statistic, ties averaged."""
    values, truth = _as_vectors(scores, labels)
    positive = truth > 0.5
    n_positive = int(positive.sum())
    n_negative = int(values.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ConfigError("a discrimination value needs both classes present")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    position = 0
    while position < values.size:
        end = position
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[position]:
            end += 1
        average_rank = 0.5 * (position + end) + 1.0
        ranks[order[position : end + 1]] = average_rank
        position = end + 1
    rank_sum = float(ranks[positive].sum())
    auc = (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    return float(auc)


def trapezoidal_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve from the empirical curve, an independent reading of the
    same quantity used to check the rank form."""
    values, truth = _as_vectors(scores, labels)
    positive = truth > 0.5
    thresholds = np.unique(np.concatenate([values, [values.max() + 1.0]]))
    tpr: list[float] = []
    fpr: list[float] = []
    for threshold in thresholds[::-1]:
        predicted = values >= threshold
        tpr.append(float((predicted & positive).sum()) / float(positive.sum()))
        fpr.append(float((predicted & ~positive).sum()) / float((~positive).sum()))
    return float(np.trapezoid(np.asarray(tpr), np.asarray(fpr)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, stepwise so ties are handled once."""
    values, truth = _as_vectors(scores, labels)
    positive = truth > 0.5
    n_positive = int(positive.sum())
    if n_positive == 0:
        raise ConfigError("an average precision needs at least one positive row")
    order = np.argsort(-values, kind="mergesort")
    sorted_positive = positive[order]
    cumulative = np.cumsum(sorted_positive)
    precision = cumulative / np.arange(1, values.size + 1)
    recall = cumulative / n_positive
    delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * delta))


@dataclass(frozen=True)
class DiscriminationReport:
    """One discrimination read-out with the row counts it was computed from."""

    auc: float
    auprc: float
    n_rows: int
    n_positive: int
    prevalence: float

    def as_dict(self) -> Mapping[str, float]:
        return {
            "auc": self.auc,
            "auprc": self.auprc,
            "n_rows": float(self.n_rows),
            "n_positive": float(self.n_positive),
            "prevalence": self.prevalence,
        }

    def exceeds(self, threshold: float) -> bool:
        return self.auc > threshold


def discrimination(scores: np.ndarray, labels: np.ndarray) -> DiscriminationReport:
    values, truth = _as_vectors(scores, labels)
    positive = int((truth > 0.5).sum())
    return DiscriminationReport(
        auc=rank_auc(values, truth),
        auprc=average_precision(values, truth),
        n_rows=int(values.size),
        n_positive=positive,
        prevalence=positive / float(values.size),
    )


def standard_error_from_ranks(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard error of the area under the curve from the rank placement.

    The quantity is the Hanley-McNeil style standard error used as the fallback when the
    paired comparison cannot be formed; the primary comparison is the paired DeLong test of
    Methods Sec. 6.10.
    """
    values, truth = _as_vectors(scores, labels)
    positive = truth > 0.5
    n_positive = int(positive.sum())
    n_negative = int(values.size - n_positive)
    if n_positive < 2 or n_negative < 2:
        raise ConfigError("a standard error needs at least two rows per class")
    auc = rank_auc(values, truth)
    area = auc * (1.0 - auc)
    first = area + (n_positive - 1) * (auc / (2.0 - auc) - auc * auc)
    second = area + (n_negative - 1) * (2.0 * auc * auc / (1.0 + auc) - auc * auc)
    variance = (first / (n_positive * n_negative) + second / (n_positive * n_negative)) / 2.0
    return float(np.sqrt(max(variance, 0.0)))


def pooled_scores(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> Mapping[str, DiscriminationReport]:
    """Discrimination per group, used for the per-region read-out of Table 3."""
    if np.asarray(scores).size != np.asarray(groups).size:
        raise ConfigError("group labels must align with the scores")
    reports: dict[str, DiscriminationReport] = {}
    for group in np.unique(groups):
        mask = groups == group
        try:
            reports[str(group)] = discrimination(scores[mask], labels[mask])
        except ConfigError:
            continue
    return reports


def low_resource_report(
    scores_by_stratum: Mapping[str, np.ndarray],
    labels_by_stratum: Mapping[str, np.ndarray],
) -> Mapping[str, DiscriminationReport]:
    """Per-stratum discrimination, which is the column the low-resource claim rests on.

    Ref: Table 1 -- the low-resource column is a separate read-out rather than a subset of
    the external one, so it is computed from the stratum that carries the smallest modality
    budget.
    """
    if set(scores_by_stratum) != set(labels_by_stratum):
        raise ConfigError("every stratum needs scores and labels")
    reports: dict[str, DiscriminationReport] = {}
    for stratum, scores in scores_by_stratum.items():
        reports[stratum] = discrimination(scores, labels_by_stratum[stratum])
    return reports
