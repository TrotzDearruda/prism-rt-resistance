"""A point estimate with an interval around it.

The same shape is reported by several parts of the study -- a prevalence rate with its exact
interval, a discrimination value with its bootstrap interval, a programme effect with its
interval head output -- so the container lives here and every subsystem imports it rather than
declaring its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Estimate, bounds, level and the rule that produced the bounds."""

    estimate: float
    lower: float
    upper: float
    level: float
    method: str

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= float(value) <= self.upper

    def shifted(self, delta: float) -> Interval:
        """Return the same interval translated, for a margin reading."""
        return Interval(
            estimate=self.estimate + delta,
            lower=self.lower + delta,
            upper=self.upper + delta,
            level=self.level,
            method=self.method,
        )

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "half_width": self.half_width,
            "level": self.level,
            "method": self.method,
        }
