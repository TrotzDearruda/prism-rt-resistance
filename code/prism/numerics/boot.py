"""Patient-level resampling.

Ref: Methods Sec. 6.10 -- bootstrap confidence intervals are drawn over patients, and
every partition is defined at the patient level, so a resample unit is a patient and
never a record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate next to the resampled interval of the same functional."""

    estimate: float
    lower: float
    upper: float
    draws: int
    level: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)


def patient_blocks(groups: Sequence[str]) -> list[np.ndarray]:
    """Index blocks of a record-level table, one block per distinct patient key."""
    order: dict[str, list[int]] = {}
    for position, key in enumerate(groups):
        order.setdefault(key, []).append(position)
    return [np.asarray(order[key], dtype=np.int64) for key in sorted(order)]


def _resample_rows(blocks: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    picks = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[int(p)] for p in picks])


def bootstrap_functional(
    functional: Callable[[np.ndarray], float],
    metric: Sequence[float],
    groups: Sequence[str],
    *,
    draws: int = 2000,
    level: float = 0.95,
    seed: int = 20240921,
) -> BootstrapResult:
    """Percentile interval for ``functional`` evaluated on patient resamples."""
    values = np.asarray(metric, dtype=np.float64)
    if values.shape[0] != len(groups):
        raise ValueError(f"metric carries {values.shape[0]} entries for {len(groups)} patient keys")
    blocks = patient_blocks(groups)
    rng = np.random.default_rng(seed)
    observed = float(functional(values))
    drawn = np.empty(draws, dtype=np.float64)
    for step in range(draws):
        rows = _resample_rows(blocks, rng)
        drawn[step] = float(functional(values[rows]))
    tail = 0.5 * (1.0 - level)
    lower = float(np.quantile(drawn, tail))
    upper = float(np.quantile(drawn, 1.0 - tail))
    return BootstrapResult(estimate=observed, lower=lower, upper=upper, draws=draws, level=level)


def model_ensemble_summary(per_seed: Sequence[float]) -> tuple[float, float]:
    """Seed-ensemble spread.

    Ref: Methods Sec. 6.10 -- variability of an estimate is the standard deviation of
    the fitted ensemble across random seeds.
    """
    values = np.asarray(per_seed, dtype=np.float64)
    if values.size == 0:
        raise ValueError("an ensemble summary needs at least one member")
    centre = float(values.mean())
    spread = float(values.std(ddof=0))
    return centre, spread
