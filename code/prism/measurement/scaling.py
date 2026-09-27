"""Scaling behaviour.

Ref: Table 7 -- the upper block reports external discrimination and programme coherence
against the fraction of the retrospective arm used for fitting, and the lower block reports
external discrimination and coherence against the fraction of records carrying spatially
resolved supervision. Ref: Results Sec. 3.2 -- the two curves behave differently, and the
caption states that neither curve is monotone to the reported precision, with the upper block
dipping slightly at 0.75 and the lower block at 0.35. The dips are reported here rather than
smoothed, so every function below returns the point where the curve turns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError


@dataclass(frozen=True)
class FitFractionPoint:
    """One row of the upper block."""

    fraction: float
    external_auc: float
    calibration_slope: float
    coherence: float
    half_width: float

    def as_dict(self) -> Mapping[str, float]:
        return {
            "fraction_of_arm": self.fraction,
            "external_auc": self.external_auc,
            "calibration_slope": self.calibration_slope,
            "programme_coherence": self.coherence,
            "pi_half_width": self.half_width,
        }


@dataclass(frozen=True)
class SupervisionFractionPoint:
    """One row of the lower block."""

    fraction: float
    external_auc: float
    coherence: float
    retention: float
    half_width: float

    def as_dict(self) -> Mapping[str, float]:
        return {
            "fraction_supervised": self.fraction,
            "external_auc": self.external_auc,
            "programme_coherence": self.coherence,
            "retention_of_named_subset": self.retention,
            "pi_half_width": self.half_width,
        }


def _validate_fractions(fractions: Sequence[float]) -> None:
    if not fractions:
        raise ConfigError("a scaling curve needs at least one fraction")
    if any(not 0.0 < value <= 1.0 for value in fractions):
        raise ConfigError("a scaling fraction must lie in (0, 1]")
    if list(fractions) != sorted(fractions):
        raise ConfigError("a scaling curve is read in increasing fraction order")


def upper_curve(points: Sequence[FitFractionPoint]) -> tuple[FitFractionPoint, ...]:
    _validate_fractions([point.fraction for point in points])
    return tuple(points)


def lower_curve(
    points: Sequence[SupervisionFractionPoint],
) -> tuple[SupervisionFractionPoint, ...]:
    _validate_fractions([point.fraction for point in points])
    return tuple(points)


def turning_points(values: Sequence[float]) -> tuple[int, ...]:
    """Positions where a curve changes direction, including the reported non-monotone dips."""
    array = np.asarray(values, dtype=np.float64)
    if array.size < 3:
        return ()
    deltas = np.diff(array)
    signs = np.sign(deltas)
    turns: list[int] = []
    for position in range(1, signs.size):
        if (
            signs[position] != 0.0
            and signs[position - 1] != 0.0
            and signs[position] != signs[position - 1]
        ):
            turns.append(position)
    return tuple(turns)


def saturation_fraction(points: Sequence[FitFractionPoint], tolerance: float) -> float:
    """Smallest fraction whose discrimination is within tolerance of the curve maximum.

    Ref: Results Sec. 3.2 -- the upper curve reaches its maximum external performance at a
    small fraction, which the manuscript reads as evidence about the geometry of the
    representation rather than about cohort size.
    """
    ordered = upper_curve(points)
    best = max(point.external_auc for point in ordered)
    for point in ordered:
        if best - point.external_auc <= tolerance:
            return point.fraction
    raise ConfigError("no fraction of the upper curve reaches the maximum within tolerance")


def lower_block_saturation(
    points: Sequence[SupervisionFractionPoint], tolerance: float
) -> Mapping[str, float]:
    """Whether the supervision curve has flattened.

    Ref: Table 7 note -- the lower block has not saturated at the available supervision
    fraction, which is reported as a limitation of the design and as the primary direction for
    extension. The function returns the evidence for that statement instead of asserting it.
    """
    ordered = lower_curve(points)
    if len(ordered) < 2:
        raise ConfigError("a saturation reading needs at least two fractions")
    gains = [
        ordered[position + 1].coherence - ordered[position].coherence
        for position in range(len(ordered) - 1)
    ]
    tail_gain = gains[-1]
    return {
        "last_coherence_gain": tail_gain,
        "sat_at_last_fraction": float(abs(tail_gain) <= tolerance),
        "auc_gain_first": ordered[1].external_auc - ordered[0].external_auc,
        "coherence_gain_total": ordered[-1].coherence - ordered[0].coherence,
    }


def gain_per_increment(points: Sequence[FitFractionPoint]) -> tuple[float, ...]:
    """Discrimination gained per step of the upper curve."""
    ordered = upper_curve(points)
    return tuple(
        ordered[position + 1].external_auc - ordered[position].external_auc
        for position in range(len(ordered) - 1)
    )


def curve_summary(
    upper: Sequence[FitFractionPoint],
    lower: Sequence[SupervisionFractionPoint],
    *,
    tolerance: float = 0.002,
) -> Mapping[str, float]:
    """The two readings the scaling section reports, in one place."""
    saturated_at = saturation_fraction(upper, tolerance)
    lower_reading = lower_block_saturation(lower, tolerance)
    return {
        "saturation_fraction": saturated_at,
        "upper_turning_points": float(len(turning_points([p.external_auc for p in upper]))),
        "lower_turning_points": float(len(turning_points([p.coherence for p in lower]))),
        "last_coherence_gain": lower_reading["last_coherence_gain"],
        "lower_saturated": lower_reading["sat_at_last_fraction"],
    }
