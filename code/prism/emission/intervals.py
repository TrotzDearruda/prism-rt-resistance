"""Interval construction and its leave-one-site-out validation.

Ref: Algorithm 1 step 16 -- the calibrated interval head is validated leave-one-site-out, so the
head is fitted on every site but one and its coverage is read on the site that was held out.
Ref: Methods Sec. 6.6 -- coverage is exact under the no-shift conditions and is measured again
under shift, and a head whose intervals narrow while its coverage falls is reported as a
coverage failure rather than as an improvement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.cohort.lexicon import Lexicon
from prism.cohort.schema import Cohort
from prism.errors import ConfigError
from prism.modelling.interval_head import z_multiplier
from prism.statistics.ci import normal_interval


@dataclass(frozen=True)
class FoldCoverage:
    """Coverage measured on one held-out site."""

    site: str
    coverage: float
    mean_width: float
    n_intervals: int

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "site": self.site,
            "coverage": self.coverage,
            "mean_width": self.mean_width,
            "n_intervals": float(self.n_intervals),
        }


def interval_from_effect(
    effect: np.ndarray, scale: np.ndarray, nominal_coverage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval around a per-programme effect."""
    centres = np.asarray(effect, dtype=np.float64)
    spreads = np.asarray(scale, dtype=np.float64)
    if centres.shape != spreads.shape:
        raise ConfigError("an effect and its scale must share a shape")
    if np.any(spreads < 0.0):
        raise ConfigError("an interval scale cannot be negative")
    half = z_multiplier(nominal_coverage) * spreads
    return centres - half, centres + half


def coverage_of(lower: np.ndarray, upper: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Empirical coverage and mean width of a block of intervals."""
    lows = np.asarray(lower, dtype=np.float64).reshape(-1)
    highs = np.asarray(upper, dtype=np.float64).reshape(-1)
    truth = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not (lows.shape == highs.shape == truth.shape):
        raise ConfigError("interval bounds and their targets must share a shape")
    if lows.size == 0:
        raise ConfigError("a coverage reading needs at least one interval")
    inside = float(np.mean((truth >= lows) & (truth <= highs)))
    width = float(np.mean(highs - lows))
    return inside, width


def leave_one_site_out_folds(
    sites: Sequence[str], per_site: Mapping[str, int]
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Row index blocks for a leave-one-site-out reading of the interval head."""
    if len(sites) < 2:
        raise ConfigError("a leave-one-site-out reading needs at least two sites")
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for site in sites:
        count = int(per_site.get(site, 0))
        offsets[site] = (cursor, cursor + count)
        cursor += count
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for site in sites:
        start, stop = offsets[site]
        held = np.arange(start, stop, dtype=np.int64)
        fit_rows = np.concatenate(
            [np.arange(*offsets[other], dtype=np.int64) for other in sites if other != site]
        )
        if held.size == 0 or fit_rows.size == 0:
            continue
        folds.append((site, fit_rows, held))
    if not folds:
        raise ConfigError("no fold carried both a fitting block and a held-out block")
    return tuple(folds)


def fold_coverages(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    site_labels: Sequence[str],
) -> tuple[FoldCoverage, ...]:
    """Coverage per site, reading the same interval block against site labels."""
    labels = np.asarray([str(label) for label in site_labels], dtype=object)
    if labels.size != np.asarray(targets).size:
        raise ConfigError("site labels must align with the interval targets")
    rows: list[FoldCoverage] = []
    for site in sorted(set(labels.tolist())):
        mask = labels == site
        coverage, width = coverage_of(
            np.asarray(lower)[mask], np.asarray(upper)[mask], np.asarray(targets)[mask]
        )
        rows.append(
            FoldCoverage(
                site=str(site),
                coverage=coverage,
                mean_width=width,
                n_intervals=int(mask.sum()),
            )
        )
    return tuple(rows)


def pooled_coverage(folds: Sequence[FoldCoverage], nominal: float) -> Mapping[str, float]:
    """Weighted coverage across folds next to the nominal level."""
    if not folds:
        raise ConfigError("a pooled coverage needs at least one fold")
    weights = np.asarray([fold.n_intervals for fold in folds], dtype=np.float64)
    coverages = np.asarray([fold.coverage for fold in folds], dtype=np.float64)
    widths = np.asarray([fold.mean_width for fold in folds], dtype=np.float64)
    pooled = float(np.sum(weights * coverages) / np.sum(weights))
    return {
        "nominal_coverage": nominal,
        "pooled_coverage": pooled,
        "coverage_gap": pooled - nominal,
        "pooled_mean_width": float(np.sum(weights * widths) / np.sum(weights)),
        "worst_fold_coverage": float(coverages.min()),
    }


def shift_reading(
    reference: Mapping[str, float], shifted: Mapping[str, float], null_band: float
) -> Mapping[str, float]:
    """Coverage degradation under shift, read against the null band.

    Ref: Table 2 caption -- the interval-head row is reported as null on discrimination and as a
    coverage failure on calibration; this is the second half of that reading.
    """
    for key in ("pooled_coverage", "pooled_mean_width"):
        if key not in reference or key not in shifted:
            raise ConfigError(f"a shift reading needs {key} on both sides")
    coverage_delta = shifted["pooled_coverage"] - reference["pooled_coverage"]
    width_delta = shifted["pooled_mean_width"] - reference["pooled_mean_width"]
    return {
        "coverage_delta": coverage_delta,
        "width_delta": width_delta,
        "coverage_failure": float(coverage_delta < -null_band),
        "narrower_intervals": float(width_delta < 0.0),
    }


def interval_arrays_from_columns(
    effects: Mapping[str, float], spreads: Mapping[str, float], nominal: float
) -> tuple[Mapping[str, tuple[float, float]], Mapping[str, float]]:
    """Per-programme intervals and their widths from named effect and scale columns."""
    if set(effects) != set(spreads):
        raise ConfigError("an effect column and its scale column must name the same programmes")
    intervals: dict[str, tuple[float, float]] = {}
    widths: dict[str, float] = {}
    for programme_id in sorted(effects):
        interval = normal_interval(effects[programme_id], spreads[programme_id], nominal)
        intervals[programme_id] = (interval.lower, interval.upper)
        widths[programme_id] = interval.width
    return intervals, widths


def cohort_site_counts(cohort: Cohort) -> Mapping[str, int]:
    """Records per site, used to build the leave-one-site-out row blocks."""
    counts: dict[str, int] = {}
    for record in cohort:
        counts[record.site] = counts.get(record.site, 0) + 1
    return counts


def programme_interval_table(lexicon: Lexicon) -> tuple[str, ...]:
    """Programme identifiers in vocabulary order, so interval columns line up with the lexicon."""
    return tuple(lexicon.programme_ids)
