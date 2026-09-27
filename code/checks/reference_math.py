"""Independent implementations of the quantities the release is verified against.

Nothing here imports ``prism``. The point of a second implementation is that it can disagree: an
area under the curve computed by counting concordant pairs, an exact binomial interval built from
the incomplete beta function, a step-down correction written as a loop over sorted values, and a
paired difference evaluated on the pairs the test would have formed. If the release's own
implementation and these agree on the same inputs, the agreement is evidence; if they were the same
code it would not be.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from scipy import special


def concordant_pairs_auc(
    scores: np.ndarray | Sequence[float], labels: np.ndarray | Sequence[int]
) -> float:
    """Area under the ROC curve by counting concordant pairs, ties counting a half."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    if values.size != truth.size:
        raise ValueError("scores and labels must share a length")
    positive = values[truth > 0.5]
    negative = values[truth <= 0.5]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes are needed")
    wins = 0.0
    for hit in positive:
        for miss in negative:
            if hit > miss:
                wins += 1.0
            elif hit == miss:
                wins += 0.5
    return wins / float(positive.size * negative.size)


def trapezoid_auc(
    scores: np.ndarray | Sequence[float], labels: np.ndarray | Sequence[int]
) -> float:
    """Area under the ROC curve by integrating the empirical curve."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    positive = truth > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("both classes are needed")
    thresholds = sorted(set(values.tolist()) | {float(values.max()) + 1.0}, reverse=True)
    xs = [0.0]
    ys = [0.0]
    for threshold in thresholds:
        predicted = values >= threshold
        xs.append(float((predicted & negative).sum()) / float(negative.sum()))
        ys.append(float((predicted & positive).sum()) / float(positive.sum()))
    area = 0.0
    for index in range(1, len(xs)):
        area += 0.5 * (xs[index] - xs[index - 1]) * (ys[index] + ys[index - 1])
    return area


def clopper_pearson_bounds(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial bounds from the incomplete beta function."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("the counts are out of range")
    alpha = 1.0 - level
    lower = (
        0.0
        if successes == 0
        else float(special.betaincinv(successes, total - successes + 1, alpha / 2.0))
    )
    upper = (
        1.0
        if successes == total
        else float(special.betaincinv(successes + 1, total - successes, 1.0 - alpha / 2.0))
    )
    return lower, upper


def positive_predictive_value(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Prevalence-weighted share of positive calls that are correct."""
    hit = sensitivity * prevalence
    miss = (1.0 - specificity) * (1.0 - prevalence)
    if hit + miss <= 0.0:
        raise ValueError("the operating point is degenerate")
    return hit / (hit + miss)


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Step-down correction written as a loop over the sorted values."""
    values = [float(value) for value in p_values]
    count = len(values)
    order = sorted(range(count), key=lambda position: (values[position], position))
    adjusted = [1.0] * count
    running = 0.0
    for rank, position in enumerate(order):
        scaled = min(1.0, values[position] * (count - rank))
        running = max(running, scaled)
        adjusted[position] = running
    return adjusted


def paired_auc_difference(
    scores_a: np.ndarray | Sequence[float],
    scores_b: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
) -> float:
    """Difference of areas on the same rows, computed from the two pair-count estimates."""
    return concordant_pairs_auc(scores_a, labels) - concordant_pairs_auc(scores_b, labels)


def bootstrap_auc_difference(
    scores_a: np.ndarray | Sequence[float],
    scores_b: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    patients: np.ndarray | Sequence[str],
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    """Patient-cluster bootstrap of the difference of two areas, hand-rolled with numpy."""
    first = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    second = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    keys = np.asarray([str(key) for key in patients], dtype=object)
    blocks: dict[str, list[int]] = {}
    for position, key in enumerate(keys.tolist()):
        blocks.setdefault(key, []).append(position)
    index_blocks = [np.asarray(rows, dtype=np.int64) for rows in blocks.values()]
    rng = np.random.default_rng(seed)
    observed = paired_auc_difference(first, second, truth)
    drawn = np.empty(draws, dtype=np.float64)
    for step in range(draws):
        picks = rng.integers(0, len(index_blocks), size=len(index_blocks))
        rows = np.concatenate([index_blocks[int(pick)] for pick in picks])
        try:
            drawn[step] = paired_auc_difference(first[rows], second[rows], truth[rows])
        except ValueError:
            drawn[step] = 0.0
    lower = float(np.quantile(drawn, 0.025))
    upper = float(np.quantile(drawn, 0.975))
    return observed, lower, upper


def fleiss_kappa(counts: np.ndarray) -> float:
    """Agreement across raters, computed from the category counts directly."""
    table = np.asarray(counts, dtype=np.float64)
    per_item = table.sum(axis=1)
    raters = per_item.mean()
    observed = float(
        np.mean(
            (np.sum((table / per_item[:, None]) ** 2, axis=1) - 1.0 / raters) / (1.0 - 1.0 / raters)
        )
    )
    marginal = table.sum(axis=0) / table.sum()
    expected = float(np.sum(marginal * marginal))
    return (observed - expected) / (1.0 - expected)


def logistic_probability(linear: float) -> float:
    return 1.0 / (1.0 + exp(-linear))


def log_loss(
    probabilities: np.ndarray | Sequence[float], labels: np.ndarray | Sequence[int]
) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    truth = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(truth * np.log(values) + (1.0 - truth) * np.log(1.0 - values)))


def calibration_slope_by_grid(
    scores: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    span: float = 6.0,
    steps: int = 601,
) -> tuple[float, float]:
    """Slope and intercept of a logistic fit, found by a grid search over the slope.

    A grid search is a different route to the same quantity than the reweighted solve the
    package uses, so agreement between the two is not automatic.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    best = (float("nan"), float("nan"), float("inf"))
    for slope in np.linspace(0.0, 3.0, steps):
        for intercept in np.linspace(-span, span, 121):
            linear = intercept + slope * values
            probabilities = 1.0 / (1.0 + np.exp(-linear))
            loss = log_loss(probabilities, truth)
            if loss < best[2]:
                best = (float(slope), float(intercept), float(loss))
    return best[0], best[1]


@dataclass(frozen=True)
class IndependentAuc:
    """Both independent readings of one area estimate, next to their gap."""

    pairs: float
    trapezoid: float

    def gap(self) -> float:
        return abs(self.pairs - self.trapezoid)

    def close(self, tolerance: float) -> bool:
        return self.gap() <= tolerance


def both_auc_readings(
    scores: np.ndarray | Sequence[float], labels: np.ndarray | Sequence[int]
) -> IndependentAuc:
    return IndependentAuc(
        pairs=concordant_pairs_auc(scores, labels),
        trapezoid=trapezoid_auc(scores, labels),
    )


def synergy_excess(alignment_cost: float, invariance_cost: float, both_cost: float) -> float:
    return float(both_cost) - (float(alignment_cost) + float(invariance_cost))


def margin_reachable(difference: float, margin: float) -> bool:
    """Whether a measured difference sits inside a declared margin."""
    return abs(float(difference)) <= float(margin)


def standard_error_of_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("a standard error needs at least two values")
    return float(array.std(ddof=1) / sqrt(array.size))
