"""Intervals for a discrimination estimate.

Ref: Methods Sec. 6.10 -- confidence intervals come from bootstrapping at the patient level, and
the variability of an estimate is the standard deviation of the fitted ensemble across random
seeds. Both routes are provided because the reported tables use the first for an interval and
the second for a spread, and mixing them would make an interval and a spread look comparable
when they are not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

from prism.errors import ConfigError
from prism.numerics.interval import Interval


def _percentile(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = 0.5 * (1.0 - level)
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def patient_cluster_bootstrap(
    values: Sequence[float],
    patients: Sequence[str],
    functional: Callable[[np.ndarray], float],
    *,
    level: float = 0.95,
    draws: int = 2000,
    seed: int = 20240921,
) -> Interval:
    """Bootstrap over patients, resampling the whole cluster of a patient at a time.

    The estimate is computed on the observed rows, and the interval on resampled patients, so a
    patient who contributed several records is resampled as one draw rather than being counted
    once per record.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    keys = np.asarray(patients, dtype=object).reshape(-1)
    if array.size != keys.size:
        raise ConfigError("a cluster bootstrap needs one patient key per value")
    if array.size == 0:
        raise ConfigError("a cluster bootstrap needs at least one value")
    blocks: dict[str, list[int]] = {}
    for position in range(array.size):
        blocks.setdefault(str(keys[position]), []).append(position)
    if len(blocks) < 2:
        raise ConfigError("a cluster bootstrap needs at least two patients")
    index_blocks = [np.asarray(rows, dtype=np.int64) for rows in blocks.values()]
    rng = np.random.default_rng(seed)
    observed = float(functional(array))
    drawn = np.empty(draws, dtype=np.float64)
    for step in range(draws):
        picks = rng.integers(0, len(index_blocks), size=len(index_blocks))
        rows = np.concatenate([index_blocks[int(pick)] for pick in picks])
        drawn[step] = float(functional(array[rows]))
    lower, upper = _percentile(drawn, level)
    return Interval(
        estimate=observed,
        lower=lower,
        upper=upper,
        level=level,
        method="patient-cluster-bootstrap",
    )


def normal_interval(estimate: float, standard_error: float, level: float = 0.95) -> Interval:
    """Wald interval, reported where a closed-form standard error is available."""
    if standard_error < 0.0:
        raise ConfigError("a standard error cannot be negative")
    quantile = float(stats.norm.ppf(1.0 - 0.5 * (1.0 - level)))
    return Interval(
        estimate=float(estimate),
        lower=float(estimate) - quantile * float(standard_error),
        upper=float(estimate) + quantile * float(standard_error),
        level=level,
        method="wald",
    )


def seed_spread(per_seed: Sequence[float]) -> tuple[float, float]:
    """Centre and spread of an estimate over the fitted ensemble.

    Ref: Methods Sec. 6.10 -- the variability of an estimate is the standard deviation across
    random seeds, taken with the population convention because the ensemble is the whole set
    that was fitted rather than a sample from one.
    """
    array = np.asarray(per_seed, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ConfigError("a seed spread needs at least one member")
    return float(array.mean()), float(array.std(ddof=0))


def interval_width_statistics(widths: Sequence[float]) -> dict[str, float]:
    """Mean, spread and extremes of a set of interval widths."""
    array = np.asarray(widths, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ConfigError("an interval width reading needs at least one width")
    return {
        "mean_width": float(array.mean()),
        "sd_width": float(array.std(ddof=0)),
        "min_width": float(array.min()),
        "max_width": float(array.max()),
    }


def delta_from_reference(interval: Interval, reference: float) -> float:
    """Where a reported interval sits relative to a declared reference value."""
    return interval.lower - float(reference)


def reference_inside(interval: Interval, reference: float) -> bool:
    return interval.contains(float(reference))
