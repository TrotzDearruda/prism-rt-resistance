"""Representation-level site and region attribution.

Ref: Table 3 Panel B -- a probe is trained to predict the care setting and the region from
the same representations the outcome read-out uses, because a well-constrained
representation should retain little site-attributable variance. The probe here is a
closed-form ridge read-out on one-hot targets, deliberately not the model's own heads, so
that the diagnostic cannot be satisfied by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError


@dataclass(frozen=True)
class LinearClassifier:
    """Ridge least-squares read-out on one-hot targets."""

    weights: np.ndarray
    intercept: np.ndarray
    n_classes: int
    ridge: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        block = _validate_block(features, self.weights.shape[0])
        scores: np.ndarray = block @ self.weights + self.intercept
        return scores

    def predict(self, features: np.ndarray) -> np.ndarray:
        predicted: np.ndarray = np.argmax(self.scores(features), axis=1)
        return predicted


def _validate_block(features: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ConfigError("a classifier must be applied to a block of the fitted width")
    return values


def fit_classifier(
    features: np.ndarray, labels: np.ndarray, n_classes: int, ridge: float = 1e-2
) -> LinearClassifier:
    """One-hot ridge regression, solved in closed form."""
    values = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels).reshape(-1)
    if values.ndim != 2:
        raise ConfigError("a classifier fit reads a two-dimensional feature block")
    if values.shape[0] != truth.size:
        raise ConfigError("every row needs one label")
    if n_classes < 2:
        raise ConfigError("a classifier needs at least two classes")
    if np.unique(truth).size < 2:
        raise ConfigError("a classifier cannot be fitted on a single-class block")
    design = np.column_stack([np.ones(values.shape[0], dtype=np.float64), values])
    targets = np.zeros((truth.size, n_classes), dtype=np.float64)
    targets[np.arange(truth.size), truth] = 1.0
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    solution = np.linalg.solve(gram, design.T @ targets)
    return LinearClassifier(
        weights=solution[1:],
        intercept=solution[0],
        n_classes=n_classes,
        ridge=ridge,
    )


def accuracy(features: np.ndarray, labels: np.ndarray, classifier: LinearClassifier) -> float:
    predicted = classifier.predict(features)
    truth = np.asarray(labels).reshape(-1)
    if predicted.size != truth.size:
        raise ConfigError("a prediction and its label block must share a length")
    return float((predicted == truth).mean())


def chance_level(labels: np.ndarray) -> float:
    """Accuracy of the majority-class rule, the reference the probe must beat."""
    truth = np.asarray(labels).reshape(-1)
    if truth.size == 0:
        raise ConfigError("a chance level needs at least one label")
    _, counts = np.unique(truth, return_counts=True)
    return float(counts.max()) / float(truth.size)


@dataclass(frozen=True)
class AttributionRow:
    """One configuration of Table 3 Panel B."""

    configuration: str
    site_accuracy: float
    region_accuracy: float
    outcome_auc: float
    site_chance: float
    region_chance: float

    def site_excess(self) -> float:
        return self.site_accuracy - self.site_chance

    def region_excess(self) -> float:
        return self.region_accuracy - self.region_chance

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "configuration": self.configuration,
            "site_probe_accuracy": self.site_accuracy,
            "region_probe_accuracy": self.region_accuracy,
            "outcome_auc_retained": self.outcome_auc,
            "site_chance_level": self.site_chance,
            "region_chance_level": self.region_chance,
        }


def attribution_row(
    configuration: str,
    features: np.ndarray,
    sites: np.ndarray,
    regions: np.ndarray,
    outcome_auc: float,
    *,
    ridge: float = 1e-2,
) -> AttributionRow:
    """Fit both probes on one representation block and read the accuracies off."""
    site_codes = _encode(sites)
    region_codes = _encode(regions)
    site_classifier = fit_classifier(features, site_codes, int(site_codes.max()) + 1, ridge)
    region_classifier = fit_classifier(features, region_codes, int(region_codes.max()) + 1, ridge)
    return AttributionRow(
        configuration=configuration,
        site_accuracy=accuracy(features, site_codes, site_classifier),
        region_accuracy=accuracy(features, region_codes, region_classifier),
        outcome_auc=outcome_auc,
        site_chance=chance_level(site_codes),
        region_chance=chance_level(region_codes),
    )


def _encode(labels: np.ndarray) -> np.ndarray:
    values = np.asarray([str(label) for label in np.asarray(labels).reshape(-1)], dtype=object)
    unique, inverse = np.unique(values, return_inverse=True)
    if unique.size < 2:
        raise ConfigError("an attribution probe needs at least two distinct labels")
    codes: np.ndarray = np.asarray(inverse, dtype=np.int64).reshape(-1)
    return codes


def attribution_table(rows: Sequence[AttributionRow]) -> tuple[AttributionRow, ...]:
    if not rows:
        raise ConfigError("an attribution table needs at least one row")
    names = [row.configuration for row in rows]
    if len(set(names)) != len(names):
        raise ConfigError("attribution configurations must be distinct")
    return tuple(rows)


def constraint_reading(rows: Sequence[AttributionRow]) -> Mapping[str, float]:
    """How far the attribution moved from the unconstrained row to the last row.

    Ref: Table 3 Panel B -- the constrained rows are read against the unconstrained one, and
    the point of the panel is that the site accuracy falls a long way while the retained
    outcome discrimination barely moves.
    """
    ordered = attribution_table(rows)
    reference = ordered[0]
    last = ordered[-1]
    return {
        "site_accuracy_drop": reference.site_accuracy - last.site_accuracy,
        "region_accuracy_drop": reference.region_accuracy - last.region_accuracy,
        "auc_retained": last.outcome_auc - reference.outcome_auc,
        "site_above_chance_after": last.site_excess(),
    }
