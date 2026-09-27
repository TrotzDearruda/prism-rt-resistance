"""The pre-specified comparison family.

Ref: Methods Sec. 6.10 -- the multiplicity family is counted beforehand and consists of five
contrasts, the Holm correction is applied inside that family alone, and no comparison outside
it is corrected. Ref: Methods Sec. 6.9 -- the family is fixed before any held-out partition is
inspected. The manuscript names the family size and the substance of the comparisons but does
not print the list as a table, so the five contrasts below are recorded as a reconstruction
with the section each one comes from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prism.errors import ConfigError

PRIMARY = "primary-equivalence"
EXTERNAL_ADVANTAGE = "external-advantage"
STRONGEST_BASELINE = "strongest-baseline-margin"
OXYGENATION_CONDITIONAL = "oxygenation-conditional-value"
SYNERGY = "component-synergy"


@dataclass(frozen=True)
class ContrastSpec:
    """One member of the family, with what it compares and where it is read."""

    name: str
    description: str
    partition: str
    reference: str
    kind: str
    paper_section: str

    def as_dict(self) -> Mapping[str, str]:
        return {
            "contrast": self.name,
            "description": self.description,
            "partition": self.partition,
            "reference": self.reference,
            "kind": self.kind,
            "paper_section": self.paper_section,
        }


FAMILY: tuple[ContrastSpec, ...] = (
    ContrastSpec(
        name=PRIMARY,
        description=(
            "constrained configuration against the equally capable unconstrained "
            "configuration on the primary partition"
        ),
        partition="internal",
        reference="constrained",
        kind="equivalence",
        paper_section="sec-6.9",
    ),
    ContrastSpec(
        name=EXTERNAL_ADVANTAGE,
        description=(
            "constrained configuration against the unconstrained configuration on the "
            "external partition"
        ),
        partition="external",
        reference="constrained",
        kind="superiority",
        paper_section="sec-6.9",
    ),
    ContrastSpec(
        name=STRONGEST_BASELINE,
        description=(
            "constrained configuration against the strongest named baseline on the "
            "external partition"
        ),
        partition="external",
        reference="constrained",
        kind="superiority",
        paper_section="sec-4.1",
    ),
    ContrastSpec(
        name=OXYGENATION_CONDITIONAL,
        description=(
            "incremental value of the programme once the oxygenation proxy is entered as a "
            "covariate"
        ),
        partition="external",
        reference="constrained",
        kind="conditional-value",
        paper_section="sec-2.1",
    ),
    ContrastSpec(
        name=SYNERGY,
        description=(
            "cost of removing both constraint terms against the sum of the individual costs"
        ),
        partition="external",
        reference="constrained",
        kind="super-additivity",
        paper_section="sec-2.2",
    ),
)

FAMILY_NAMES: tuple[str, ...] = tuple(spec.name for spec in FAMILY)


def family() -> tuple[ContrastSpec, ...]:
    return FAMILY


def spec_for(name: str) -> ContrastSpec:
    for candidate in FAMILY:
        if candidate.name == name:
            return candidate
    raise ConfigError(f"no pre-specified contrast is named {name!r}")


def validate_family(specs: Sequence[ContrastSpec] = FAMILY) -> None:
    """Check the family against the size the analysis plan fixed."""
    if len(specs) != 5:
        raise ConfigError(
            f"the analysis plan fixes five contrasts, this family carries {len(specs)}"
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ConfigError("contrast names inside the family must be distinct")


def ordered_p_values(p_values: Mapping[str, float]) -> tuple[float, ...]:
    """Raw p-values in the declared family order, refusing a partial family."""
    missing = [spec.name for spec in FAMILY if spec.name not in p_values]
    if missing:
        raise ConfigError(f"the family is missing p-values for {missing}")
    extra = [name for name in p_values if name not in FAMILY_NAMES]
    if extra:
        raise ConfigError(
            f"{extra} sit outside the pre-specified family and are reported as exploratory"
        )
    return tuple(float(p_values[spec.name]) for spec in FAMILY)


def synergy_statistic(
    cost_alignment: float, cost_invariance: float, cost_both: float
) -> Mapping[str, float]:
    """Super-additivity of the two constraint terms.

    Ref: Results Sec. 2.2 -- removing the alignment term alone costs some of the external
    partition, removing the invariance term alone costs more, and removing both costs more than
    the sum of the two individual costs; the excess is the quantity that carries the
    architectural reading.
    """
    additive = float(cost_alignment) + float(cost_invariance)
    return {
        "cost_alignment": float(cost_alignment),
        "cost_invariance": float(cost_invariance),
        "cost_both": float(cost_both),
        "additive_expectation": additive,
        "super_additive_excess": float(cost_both) - additive,
        "interchangeable": float(abs(float(cost_both) - additive) <= 1e-12),
    }


def conditional_value_reading(
    gain_with_proxy: Mapping[str, float], null_band: float, retained: Sequence[str]
) -> Mapping[str, float]:
    """Incremental value per programme once the proxy is in the model.

    Ref: Results Sec. 2.1 -- the conditional test is read as several programmes at once rather
    than as one test, so the reading reports per-programme gains next to how many of the named
    retained programmes cleared the band.
    """
    if not gain_with_proxy:
        raise ConfigError("a conditional reading needs at least one programme gain")
    cleared = [name for name in retained if gain_with_proxy.get(name, 0.0) > null_band]
    return {
        "n_programmes": float(len(gain_with_proxy)),
        "n_cleared": float(len(cleared)),
        "retained_cleared": float(len(cleared)) / float(len(retained)) if retained else 0.0,
        "min_gain": float(min(gain_with_proxy.values())),
        "max_gain": float(max(gain_with_proxy.values())),
    }
