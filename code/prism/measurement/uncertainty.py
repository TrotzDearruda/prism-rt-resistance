"""Prediction-interval bookkeeping.

Ref: Methods Sec. 6.10 -- every table carries the numerical variability of its estimates:
the final row of the main table gives the mean prediction-interval half-width per column,
the cells of the transfer matrix carry their own interval, and the table notes repeat the
half-widths so a reader can size an effect against the interval without a lookup.
Ref: Table 8 -- the ensemble comparison is read as a mean interval width next to an
empirical coverage, which is why the two quantities are reported together here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError

INTERVAL_LEVEL = 0.90


@dataclass(frozen=True)
class ColumnVariability:
    """One column's half-width and the count it was computed from."""

    column: str
    half_width: float
    n_values: int
    level: float = INTERVAL_LEVEL

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "column": self.column,
            "half_width": self.half_width,
            "n_values": float(self.n_values),
            "level": self.level,
        }


def half_width(values: Sequence[float], level: float = INTERVAL_LEVEL) -> float:
    """Half-width of a percentile interval, the quantity printed in the table footer."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ConfigError("a half-width needs at least one value")
    tail = 0.5 * (1.0 - level)
    lower = float(np.quantile(array, tail))
    upper = float(np.quantile(array, 1.0 - tail))
    return 0.5 * (upper - lower)


def half_width_table(columns: Mapping[str, Sequence[float]]) -> tuple[ColumnVariability, ...]:
    """The footer row of a table: one half-width per column, in the order given."""
    return tuple(
        ColumnVariability(column=name, half_width=half_width(values), n_values=len(values))
        for name, values in columns.items()
    )


@dataclass(frozen=True)
class EnsembleContrast:
    """One row of the ensemble comparison: discrimination, calibration and interval."""

    configuration: str
    external_auc: float
    calibration_slope: float
    mean_interval_width: float
    empirical_coverage: float

    def width_reduction(self, reference: EnsembleContrast) -> float:
        return reference.mean_interval_width - self.mean_interval_width

    def coverage_gain(self, reference: EnsembleContrast) -> float:
        return self.empirical_coverage - reference.empirical_coverage

    def auc_gain(self, reference: EnsembleContrast) -> float:
        return self.external_auc - reference.external_auc

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "configuration": self.configuration,
            "external_auc": self.external_auc,
            "calibration_slope": self.calibration_slope,
            "mean_interval_width": self.mean_interval_width,
            "empirical_coverage": self.empirical_coverage,
        }


def ensemble_table(rows: Sequence[EnsembleContrast]) -> tuple[EnsembleContrast, ...]:
    """Validate and order the ensemble rows as the table reports them."""
    if not rows:
        raise ConfigError("an ensemble table needs at least one row")
    names = [row.configuration for row in rows]
    if len(set(names)) != len(names):
        raise ConfigError("ensemble configurations must be distinct")
    return tuple(rows)


def aggregation_reading(rows: Sequence[EnsembleContrast]) -> Mapping[str, float]:
    """Which quantity the ensembling moved.

    Ref: Results Sec. 3.3 -- aggregation across random seeds moves discrimination slightly
    while helping calibration and interval width; the regime ensemble moves discrimination
    instead. The function reports the two margins so the reading is not asserted in prose.
    """
    ordered = ensemble_table(rows)
    baseline = ordered[0]
    last = ordered[-1]
    return {
        "auc_margin": last.auc_gain(baseline),
        "calibration_margin": last.calibration_slope - baseline.calibration_slope,
        "width_margin": last.width_reduction(baseline),
        "coverage_margin": last.coverage_gain(baseline),
    }
