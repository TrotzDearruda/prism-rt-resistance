"""Decision-curve analysis.

Ref: Table 1 -- clinical usefulness is reported as decision-curve net benefit at a
pre-specified threshold band, and Results Sec. 2.1 groups it with calibration rather than
with discrimination because the quantity exists to serve a threshold decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError

DEFAULT_THRESHOLD_BAND: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)


def net_benefit(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Net benefit of acting on the probabilities at one threshold."""
    if not 0.0 < threshold < 1.0:
        raise ConfigError("a decision threshold must lie strictly inside (0, 1)")
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size != truth.size or values.size == 0:
        raise ConfigError("net benefit needs aligned probabilities and labels")
    predicted = values >= threshold
    true_positive = float((predicted & (truth > 0.5)).sum())
    false_positive = float((predicted & (truth <= 0.5)).sum())
    total = float(values.size)
    odds = threshold / (1.0 - threshold)
    return true_positive / total - false_positive / total * odds


def net_benefit_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_BAND,
) -> Mapping[float, float]:
    return {float(t): net_benefit(probabilities, labels, float(t)) for t in thresholds}


@dataclass(frozen=True)
class DecisionReport:
    """Net benefit across the band with the treat-all and treat-none references."""

    band: tuple[float, ...]
    model: Mapping[float, float]
    treat_all: Mapping[float, float]
    treat_none: Mapping[float, float]

    def mean_benefit(self) -> float:
        return float(np.mean([self.model[t] for t in self.band]))

    def best_threshold(self) -> float:
        return max(self.band, key=lambda t: self.model[t])

    def advantage_over_treat_all(self, threshold: float) -> float:
        return self.model[threshold] - self.treat_all[threshold]

    def as_dict(self) -> Mapping[str, float]:
        values: dict[str, float] = {"mean_net_benefit": self.mean_benefit()}
        for threshold in self.band:
            values[f"net_benefit_at_{threshold:.2f}"] = self.model[threshold]
        return values


def decision_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_BAND,
) -> DecisionReport:
    """Net benefit for the model next to the two trivial strategies."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    band = tuple(float(t) for t in thresholds)
    model: dict[float, float] = {}
    treat_all: dict[float, float] = {}
    treat_none: dict[float, float] = {}
    for threshold in band:
        model[threshold] = net_benefit(values, truth, threshold)
        treat_all[threshold] = net_benefit(np.ones_like(values), truth, threshold)
        treat_none[threshold] = 0.0
    return DecisionReport(band=band, model=model, treat_all=treat_all, treat_none=treat_none)


def clinical_usefulness_margin(
    candidate: DecisionReport, reference: DecisionReport
) -> Mapping[str, float]:
    """Where the candidate beats the reference across the band.

    Ref: Table 1 -- the net-benefit column is read as a margin against the comparator named
    in the row, not as an absolute number, so the comparison is returned per threshold.
    """
    if candidate.band != reference.band:
        raise ConfigError("two decision reports can only be compared on one band")
    margins: dict[str, float] = {}
    for threshold in candidate.band:
        margins[f"margin_at_{threshold:.2f}"] = (
            candidate.model[threshold] - reference.model[threshold]
        )
    margins["mean_margin"] = candidate.mean_benefit() - reference.mean_benefit()
    return margins


def optimal_threshold_band(prevalence: float, span: float = 0.10) -> tuple[float, ...]:
    """Band centred on the deployment prevalence, the operating region a pathway uses.

    Ref: Results Sec. 2.1 -- the reported interval fixes the prevalence at which the test is
    read, so the band is anchored there rather than on a fixed grid.
    """
    if not 0.0 < prevalence < 1.0:
        raise ConfigError("a prevalence must lie strictly inside (0, 1)")
    low = max(0.01, prevalence - span / 2.0)
    high = min(0.99, prevalence + span / 2.0)
    return tuple(float(value) for value in np.linspace(low, high, 5))
