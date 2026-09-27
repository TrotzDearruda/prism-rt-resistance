"""Calibration.

Ref: Table 1 -- calibration is reported as a slope and an intercept alongside the
discrimination column, and the table's own reading is that the baselines are over-confident
because their slopes sit below one. Ref: Results Sec. 2.1 -- the decision-curve net benefit
is grouped with calibration rather than with discrimination, because a threshold decision
depends on where the probabilities sit and not only on their ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError

MAX_ITERATIONS = 100
CONVERGENCE = 1e-9


def logit(probabilities: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    transformed: np.ndarray = np.log(clipped) - np.log1p(-clipped)
    return transformed


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Logistic transform, evaluated on a clipped exponent.

    The clip is at the point where the transform is already at its floating-point limit, so it
    changes no reported value while keeping the exponential inside range; without it a large
    negative score raises an overflow warning on the branch that is discarded anyway.
    """
    limited = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    output: np.ndarray = 1.0 / (1.0 + np.exp(-limited))
    return output


@dataclass(frozen=True)
class CalibrationReport:
    """Slope, intercept, Brier score and the counts behind them."""

    slope: float
    intercept: float
    brier: float
    n_rows: int
    n_positive: int

    def over_confident(self) -> bool:
        return self.slope < 1.0

    def as_dict(self) -> Mapping[str, float]:
        return {
            "calibration_slope": self.slope,
            "calibration_intercept": self.intercept,
            "brier": self.brier,
            "n_rows": float(self.n_rows),
            "n_positive": float(self.n_positive),
        }


def fit_logistic(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Iteratively reweighted least squares on a two-column design."""
    values = np.asarray(design, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != truth.size:
        raise ConfigError("a logistic fit needs a row per target")
    coefficients = np.zeros(values.shape[1], dtype=np.float64)
    for _ in range(MAX_ITERATIONS):
        linear = values @ coefficients
        probabilities = sigmoid(linear)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = values.T @ (truth - probabilities)
        hessian = values.T @ (values * weights[:, None])
        hessian = hessian + 1e-8 * np.eye(hessian.shape[0], dtype=np.float64)
        step = np.linalg.solve(hessian, gradient)
        coefficients = coefficients + step
        if float(np.max(np.abs(step))) < CONVERGENCE:
            break
    return coefficients


def calibration_report(scores: np.ndarray, labels: np.ndarray) -> CalibrationReport:
    """Slope and intercept of the outcome regressed on the predicted logit."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size != truth.size or values.size == 0:
        raise ConfigError("a calibration report needs aligned scores and labels")
    if len(np.unique(truth > 0.5)) < 2:
        raise ConfigError("a calibration slope needs both classes present")
    probabilities = sigmoid(values)
    design = np.column_stack([np.ones(values.size, dtype=np.float64), values])
    coefficients = fit_logistic(design, truth)
    delta = probabilities - truth
    return CalibrationReport(
        slope=float(coefficients[1]),
        intercept=float(coefficients[0]),
        brier=float(np.mean(delta * delta)),
        n_rows=int(values.size),
        n_positive=int((truth > 0.5).sum()),
    )


def calibration_from_probabilities(
    probabilities: np.ndarray, labels: np.ndarray
) -> CalibrationReport:
    return calibration_report(logit(probabilities), labels)


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned observed rate against predicted rate, with the bin counts returned too."""
    if bins < 2:
        raise ConfigError("a reliability curve needs at least two bins")
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size != truth.size:
        raise ConfigError("a reliability curve needs aligned probabilities and labels")
    edges = np.linspace(0.0, 1.0, bins + 1)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for position in range(bins):
        lower, upper = edges[position], edges[position + 1]
        mask = (values >= lower) & (values < upper if position < bins - 1 else values <= upper)
        if not mask.any():
            continue
        predicted.append(float(values[mask].mean()))
        observed.append(float(truth[mask].mean()))
        counts.append(float(mask.sum()))
    return (
        np.asarray(predicted, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Weighted gap between the observed rate and the predicted rate."""
    predicted, observed, counts = reliability_curve(probabilities, labels, bins)
    if counts.sum() == 0.0:
        raise ConfigError("no bin carried a row")
    weights = counts / counts.sum()
    gap = np.abs(predicted - observed)
    return float(np.sum(weights * gap))


def slope_gap_from_one(report: CalibrationReport) -> float:
    """Distance of the slope from the perfect-calibration value.

    Ref: Results Sec. 2.1 -- the baselines sit below one; the constraining cost is reported
    as a distance from one rather than as a verdict.
    """
    return 1.0 - report.slope
