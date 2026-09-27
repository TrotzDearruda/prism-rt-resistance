"""Prevalence, rule-in arithmetic and the interval reported for the response rate.

Ref: Results Sec. 2.1 -- the pathological complete response rate in the retrospective arm
is 19.4 per cent with a 95 per cent interval of [18.2, 20.6]; that interval fixes the
prevalence at which a positive predictive value must be read, and rule-in statements are
therefore reported as positive predictive values rather than as sensitivities.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from prism.errors import CohortError
from prism.numerics.interval import Interval

REPORTED_PREVALENCE = 0.194
REPORTED_PREVALENCE_CI = (0.182, 0.206)
DEPLOYMENT_PREVALENCE = REPORTED_PREVALENCE


def clopper_pearson(successes: int, total: int, level: float = 0.95) -> Interval:
    """Exact binomial interval, the conservative choice for a low event rate."""
    if total <= 0:
        raise CohortError("an interval needs a positive denominator")
    if not 0 <= successes <= total:
        raise CohortError("successes must lie between zero and the denominator")
    alpha = 1.0 - level
    estimate = successes / total
    if successes == 0:
        lower = 0.0
    else:
        lower = float(stats.beta.ppf(alpha / 2.0, successes, total - successes + 1))
    if successes == total:
        upper = 1.0
    else:
        upper = float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return Interval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        level=level,
        method="clopper-pearson",
    )


def wilson(proportion: float, total: int, level: float = 0.95) -> Interval:
    """Score interval, reported alongside the exact one when the design is unbalanced."""
    if total <= 0:
        raise CohortError("an interval needs a positive denominator")
    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * float(np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)))
        / denominator
    )
    return Interval(
        estimate=proportion,
        lower=max(0.0, centre - half),
        upper=min(1.0, centre + half),
        level=level,
        method="wilson",
    )


def reported_prevalence_interval() -> Interval:
    """The interval printed in Results Sec. 2.1, reconstructed from the reported counts."""
    total = 3847
    successes = int(round(REPORTED_PREVALENCE * total))
    return clopper_pearson(successes, total, level=0.95)


def prevalence_matches_report(tolerance: float = 1e-3) -> bool:
    """Whether the reconstructed interval reproduces the printed bounds.

    The check is reported as a mismatch when it fails rather than being forced to agree: the
    printed bounds come from the study's own arm, while the reconstruction here uses the
    held-out record count stated in Results Sec. 2.1.
    """
    interval = reported_prevalence_interval()
    return (
        abs(interval.estimate - REPORTED_PREVALENCE) <= tolerance
        and abs(interval.lower - REPORTED_PREVALENCE_CI[0]) <= tolerance
        and abs(interval.upper - REPORTED_PREVALENCE_CI[1]) <= tolerance
    )


def reconstruction_gap() -> tuple[float, float]:
    """Signed gap between the reconstructed bounds and the printed ones."""
    interval = reported_prevalence_interval()
    return (
        interval.lower - REPORTED_PREVALENCE_CI[0],
        interval.upper - REPORTED_PREVALENCE_CI[1],
    )
