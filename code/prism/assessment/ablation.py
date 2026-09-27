"""Ablation matrix and tier-zero controls.

Ref: Table 2 -- every row removes or replaces one component of the objective of Eq. (1). Two of
the rows are tier-zero controls: the alignment term is fed a permuted metabolite matrix, and the
invariance penalty is fed randomised site labels. What makes those two controls meaningful is
stated in Results Sec. 2.2: both return the result to an unconstrained state, which is the
evidence that the two computed terms are doing something rather than acting as arbitrary
regularisers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from prism.errors import ConfigError
from prism.objectives.total import ObjectiveWeights
from prism.provenance.config import ExperimentConfig


@dataclass(frozen=True)
class AblationArm:
    """One row of the ablation table, expressed as a change to the run configuration."""

    key: str
    arm: str
    description: str
    interpretable: str
    zeroes_alignment: bool
    zeroes_invariance: bool
    drops_decoder: bool
    drops_interval: bool
    permutes_metabolites: bool
    randomises_sites: bool

    def is_tier_zero(self) -> bool:
        return self.permutes_metabolites or self.randomises_sites

    def as_dict(self) -> Mapping[str, object]:
        return {
            "key": self.key,
            "arm": self.arm,
            "description": self.description,
            "programme_interpretable": self.interpretable,
            "tier_zero_control": self.is_tier_zero(),
        }


ABLATIONS: tuple[AblationArm, ...] = (
    AblationArm(
        key="full_prism_constrained",
        arm="constrained",
        description="the complete objective of Eq. (1) under the feasibility constraint",
        interpretable="yes",
        zeroes_alignment=False,
        zeroes_invariance=False,
        drops_decoder=False,
        drops_interval=False,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
    AblationArm(
        key="minus_alignment",
        arm="constrained",
        description="the alignment term removed from the objective",
        interpretable="partial",
        zeroes_alignment=True,
        zeroes_invariance=False,
        drops_decoder=False,
        drops_interval=False,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
    AblationArm(
        key="minus_invariance",
        arm="constrained",
        description="the invariance penalty removed from the objective",
        interpretable="partial",
        zeroes_alignment=False,
        zeroes_invariance=True,
        drops_decoder=False,
        drops_interval=False,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
    AblationArm(
        key="minus_sparse_signed_decoder",
        arm="backbone-only",
        description="the sparse signed decoder replaced by a full-risk head",
        interpretable="no",
        zeroes_alignment=False,
        zeroes_invariance=False,
        drops_decoder=True,
        drops_interval=False,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
    AblationArm(
        key="minus_interval_head",
        arm="no-interval",
        description="the interval head removed, leaving the point estimate only",
        interpretable="yes",
        zeroes_alignment=False,
        zeroes_invariance=False,
        drops_decoder=False,
        drops_interval=True,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
    AblationArm(
        key="minus_alignment_and_invariance",
        arm="constrained",
        description="both constraint terms removed at once",
        interpretable="no",
        zeroes_alignment=True,
        zeroes_invariance=True,
        drops_decoder=False,
        drops_interval=False,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
    AblationArm(
        key="tier0_permuted_metabolite_alignment",
        arm="constrained",
        description="the alignment term fed a permuted metabolite matrix",
        interpretable="no",
        zeroes_alignment=False,
        zeroes_invariance=False,
        drops_decoder=False,
        drops_interval=False,
        permutes_metabolites=True,
        randomises_sites=False,
    ),
    AblationArm(
        key="tier0_randomised_site_labels",
        arm="constrained",
        description="the invariance penalty fed randomised site labels",
        interpretable="partial",
        zeroes_alignment=False,
        zeroes_invariance=False,
        drops_decoder=False,
        drops_interval=False,
        permutes_metabolites=False,
        randomises_sites=True,
    ),
    AblationArm(
        key="backbone_only_no_constraint",
        arm="backbone-only",
        description="the pretrained representation read without any constrained component",
        interpretable="no",
        zeroes_alignment=False,
        zeroes_invariance=False,
        drops_decoder=True,
        drops_interval=True,
        permutes_metabolites=False,
        randomises_sites=False,
    ),
)


def ablation_arms() -> tuple[AblationArm, ...]:
    return ABLATIONS


def arm_for(key: str) -> AblationArm:
    for arm in ABLATIONS:
        if arm.key == key:
            return arm
    raise ConfigError(f"no ablation row is named {key!r}")


def configuration_for(config: ExperimentConfig, key: str) -> ExperimentConfig:
    """Apply one ablation row to a base configuration."""
    arm = arm_for(key)
    weights = ObjectiveWeights(
        lambda_alignment=0.0 if arm.zeroes_alignment else config.objective.lambda_alignment,
        lambda_invariance=0.0 if arm.zeroes_invariance else config.objective.lambda_invariance,
        lambda_programme=config.objective.lambda_programme,
        lambda_sign=config.objective.lambda_sign,
    )
    objective = replace(
        config.objective,
        lambda_alignment=weights.lambda_alignment,
        lambda_invariance=weights.lambda_invariance,
    )
    return replace(config, arm=arm.arm, objective=objective, name=f"{config.name}::{key}")


def permute_metabolite_rows(
    metabolite: np.ndarray, *, seed: int, within_region: bool = True, regions: Sequence[str] = ()
) -> np.ndarray:
    """Break the pairing between a region representation and its metabolite block.

    The permutation is taken inside each region when region labels are supplied, so the tier-zero
    control removes the row pairing without also removing the region-level differences that the
    alignment term could otherwise pick up from the batch composition.
    """
    values = np.asarray(metabolite, dtype=np.float64)
    if values.ndim != 2:
        raise ConfigError("a metabolite block is rows by vocabulary entries")
    rng = np.random.default_rng(seed)
    if not within_region or not regions:
        order = rng.permutation(values.shape[0])
        return values[order]
    labels = np.asarray([str(label) for label in regions], dtype=object)
    if labels.size != values.shape[0]:
        raise ConfigError("region labels must align with the metabolite rows")
    permuted = values.copy()
    for region in sorted(set(labels.tolist())):
        mask = np.flatnonzero(labels == region)
        if mask.size < 2:
            continue
        order = rng.permutation(mask.size)
        permuted[mask] = values[mask[order]]
    return permuted


def randomise_site_labels(sites: np.ndarray, *, seed: int) -> np.ndarray:
    """Shuffle site labels, keeping the label histogram intact."""
    values = np.asarray(sites).reshape(-1)
    rng = np.random.default_rng(seed)
    shuffled: np.ndarray = values[rng.permutation(values.size)]
    return shuffled


def control_recovery(
    full_value: float, control_value: float, unconstrained_value: float
) -> Mapping[str, float]:
    """How far a tier-zero control moved the result back towards the unconstrained state.

    Ref: Results Sec. 2.2 -- the controls are read by whether they return the result to the
    unconstrained state, so the quantity reported is the share of the constraint's effect that the
    control removed.
    """
    effect = float(full_value) - float(unconstrained_value)
    if abs(effect) < 1e-12:
        return {"constraint_effect": 0.0, "recovered_share": 0.0}
    recovered = (float(full_value) - float(control_value)) / effect
    return {"constraint_effect": effect, "recovered_share": recovered}


def table_rows(
    measured: Mapping[str, Sequence[float]],
) -> tuple[Mapping[str, object], ...]:
    """Assemble the ablation table, requiring a row for every declared arm."""
    missing = [arm.key for arm in ABLATIONS if arm.key not in measured]
    if missing:
        raise ConfigError(f"the ablation table is missing measured rows for {missing}")
    rows: list[Mapping[str, object]] = []
    for arm in ABLATIONS:
        values = list(measured[arm.key])
        if len(values) != 3:
            raise ConfigError(
                f"ablation row {arm.key} needs internal, external and lowest-resource values"
            )
        rows.append(
            {
                "configuration": arm.key,
                "internal_auc": float(values[0]),
                "external_auc": float(values[1]),
                "lowest_resource_auc": float(values[2]),
                "programme_interpretable": arm.interpretable,
                "tier_zero_control": arm.is_tier_zero(),
            }
        )
    return tuple(rows)


def interpretability_reading(rows: Sequence[Mapping[str, object]]) -> Mapping[str, float]:
    """Count of rows that still emit an interpretable programme."""
    if not rows:
        raise ConfigError("an interpretability reading needs at least one row")
    yes = sum(1 for row in rows if row.get("programme_interpretable") == "yes")
    partial = sum(1 for row in rows if row.get("programme_interpretable") == "partial")
    return {
        "n_rows": float(len(rows)),
        "n_interpretable": float(yes),
        "n_partial": float(partial),
        "interpretable_fraction": yes / float(len(rows)),
    }
