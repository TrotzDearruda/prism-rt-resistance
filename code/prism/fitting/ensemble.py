"""Seed and regime ensembling.

Ref: Table 8 -- four configurations are compared: a single initialisation, a five-member
ensemble across random seeds, a ten-member ensemble across random seeds, and a regime ensemble
that pools the probe with the constrained configuration. Ref: Results Sec. 3.3 -- seed
aggregation moves discrimination only slightly while helping calibration and interval width, and
the regime ensemble is reported as an auxiliary comparison because its gain sits on
discrimination instead; the primary results use the seed ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError
from prism.measurement.calibration import calibration_from_probabilities
from prism.measurement.discrimination import rank_auc
from prism.measurement.uncertainty import EnsembleContrast
from prism.numerics.seed import ensemble_seeds, set_seed


@dataclass(frozen=True)
class EnsembleMember:
    """One fitted member and the scores it produced on the evaluation cohort."""

    name: str
    seed: int
    probabilities: np.ndarray

    def validate(self, n_rows: int) -> None:
        if self.probabilities.shape[0] != n_rows:
            raise ConfigError(
                f"member {self.name} carries {self.probabilities.shape[0]} scores, expected "
                f"{n_rows}"
            )


@dataclass(frozen=True)
class SeedEnsemble:
    """Members sharing one recipe but differing in initialisation."""

    members: tuple[EnsembleMember, ...]

    def n_members(self) -> int:
        return len(self.members)

    def pooled(self) -> np.ndarray:
        """Mean probability across members, the aggregation the tables report."""
        if not self.members:
            raise ConfigError("an ensemble needs at least one member")
        n_rows = self.members[0].probabilities.shape[0]
        for member in self.members:
            member.validate(n_rows)
        stacked = np.stack([member.probabilities for member in self.members], axis=0)
        pooled: np.ndarray = stacked.mean(axis=0)
        return pooled

    def spread(self) -> np.ndarray:
        """Per-row standard deviation across members."""
        if not self.members:
            raise ConfigError("an ensemble needs at least one member")
        stacked = np.stack([member.probabilities for member in self.members], axis=0)
        spread: np.ndarray = stacked.std(axis=0, ddof=0)
        return spread


def build_seed_ensemble(
    probabilities: Sequence[np.ndarray], base_seed: int, prefix: str = "member"
) -> SeedEnsemble:
    """Assemble a seed ensemble, tagging each member with the seed that produced it."""
    if not probabilities:
        raise ConfigError("an ensemble needs at least one member")
    seeds = ensemble_seeds(len(probabilities), base=base_seed)
    members = tuple(
        EnsembleMember(name=f"{prefix}{position:02d}", seed=seed, probabilities=np.asarray(values))
        for position, (seed, values) in enumerate(zip(seeds, probabilities, strict=True))
    )
    return SeedEnsemble(members=members)


def regime_ensemble(
    probe_probabilities: np.ndarray,
    constrained_probabilities: np.ndarray,
    *,
    probe_weight: float = 0.4,
) -> np.ndarray:
    """Pool the frozen-probe regime with the constrained regime.

    Ref: Results Sec. 3.3 -- this mixture is reported for the auxiliary comparison rather than
    for the primary result, because its gain is in discrimination rather than in calibration.
    """
    if not 0.0 <= probe_weight <= 1.0:
        raise ConfigError("a regime weight must lie in [0, 1]")
    first = np.asarray(probe_probabilities, dtype=np.float64).reshape(-1)
    second = np.asarray(constrained_probabilities, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ConfigError("a regime ensemble pools two score vectors of one length")
    pooled: np.ndarray = probe_weight * first + (1.0 - probe_weight) * second
    return pooled


def contrast_row(
    configuration: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    interval_targets: np.ndarray,
) -> EnsembleContrast:
    """Read one configuration into the columns of the ensemble table.

    Discrimination and calibration come from the response scores, and the interval columns come
    from the interval head's output against the held-out programme effects. The two are carried
    together in one row because the study reads them together: a configuration that widens its
    intervals while keeping its coverage is reported differently from one that narrows them and
    loses it.
    """
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).reshape(-1)
    if values.shape != truth.shape:
        raise ConfigError("a contrast row needs one score per label")
    lower = np.asarray(interval_lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(interval_upper, dtype=np.float64).reshape(-1)
    targets = np.asarray(interval_targets, dtype=np.float64).reshape(-1)
    if not (lower.shape == upper.shape == targets.shape):
        raise ConfigError("interval bounds and their targets must share a length")
    if lower.size == 0:
        raise ConfigError("a contrast row needs at least one interval")
    calibration = calibration_from_probabilities(values, truth)
    inside = float(np.mean((targets >= lower) & (targets <= upper)))
    return EnsembleContrast(
        configuration=configuration,
        external_auc=rank_auc(values, truth),
        calibration_slope=calibration.slope,
        mean_interval_width=float(np.mean(upper - lower)),
        empirical_coverage=inside,
    )


def member_agreement(ensemble: SeedEnsemble, labels: np.ndarray) -> Mapping[str, float]:
    """How much the members agree with each other and with the pooled reading."""
    truth = np.asarray(labels).reshape(-1)
    pooled = ensemble.pooled()
    if pooled.shape != truth.shape:
        raise ConfigError("an agreement reading needs one label per pooled score")
    member_aucs = np.asarray(
        [rank_auc(member.probabilities, truth) for member in ensemble.members],
        dtype=np.float64,
    )
    return {
        "pooled_auc": rank_auc(pooled, truth),
        "member_auc_mean": float(member_aucs.mean()),
        "member_auc_sd": float(member_aucs.std(ddof=0)),
        "member_auc_min": float(member_aucs.min()),
        "member_auc_max": float(member_aucs.max()),
        "mean_row_spread": float(ensemble.spread().mean()),
    }


def reseed_for_member(base_seed: int, member: int) -> int:
    """Seed a member takes, so a table can be rebuilt member by member."""
    set_seed(base_seed + 101 * member)
    return base_seed + 101 * member


def choose_configuration(
    single: EnsembleContrast,
    seed_ensembles: Sequence[EnsembleContrast],
    regime: EnsembleContrast,
    *,
    primary_metric: str = "calibration_slope",
) -> str:
    """Which configuration the primary results use.

    Ref: Results Sec. 3.3 -- the seed ensemble is used for the primary results because it
    improves the quantity a threshold decision depends on, and the regime ensemble is kept for
    the auxiliary comparison. The choice is made on the named metric rather than asserted, so a
    cohort where the ordering reverses is reported rather than hidden.
    """
    candidates = (single, *seed_ensembles)
    best_seed = max(candidates, key=lambda row: _metric(row, primary_metric))
    regime_value = _metric(regime, primary_metric)
    if regime_value > _metric(best_seed, primary_metric):
        return regime.configuration
    return best_seed.configuration


def _metric(row: EnsembleContrast, name: str) -> float:
    if name == "calibration_slope":
        return row.calibration_slope
    if name == "external_auc":
        return row.external_auc
    if name == "mean_interval_width":
        return -row.mean_interval_width
    if name == "empirical_coverage":
        return row.empirical_coverage
    raise ConfigError(f"the ensemble table has no column named {name!r}")
