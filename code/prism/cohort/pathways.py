"""Programme-level statements the emission is checked against.

Ref: Results Sec. 2.1 and Discussion Sec. 4.1 -- after the oxygenation proxy enters the
model, the glycolytic, glutaminolytic and lipid-desaturation programmes keep incremental
value, the hypoxia-linked redox programme loses all of it, and the remaining two are too
small or inconsistent to be conclusive. Ref: Table 7 -- programme coherence is the
agreement between the emitted programme and the metabolite covariance structure of the
held-out supervised region.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prism.errors import CohortError
from prism.notation import PROGRAMME_NAMES

RETAINS = "retains-incremental-value"
LOSES = "loses-conditional-value"
INCONCLUSIVE = "inconclusive"

CONDITIONAL_RETENTION: Mapping[str, str] = {
    "P1": RETAINS,
    "P2": RETAINS,
    "P3": RETAINS,
    "P4": LOSES,
    "P5": INCONCLUSIVE,
    "P6": INCONCLUSIVE,
}

CONDITIONAL_LABEL: Mapping[str, str] = {
    RETAINS: "keeps incremental value once the oxygenation proxy is present",
    LOSES: "carries no incremental value once the oxygenation proxy is present",
    INCONCLUSIVE: "is neither large nor consistent enough to be conclusive",
}


@dataclass(frozen=True)
class ProgrammeTarget:
    """What the study states about one programme."""

    programme_id: str
    name: str
    conditional_retention: str
    block_weight: float
    expected_sign_pattern: tuple[int, ...]

    def retains(self) -> bool:
        return self.conditional_retention == RETAINS

    def label(self) -> str:
        return CONDITIONAL_LABEL[self.conditional_retention]


def programme_targets(
    weights: tuple[float, ...], sign_patterns: tuple[tuple[int, ...], ...]
) -> tuple[ProgrammeTarget, ...]:
    if len(weights) != len(PROGRAMME_NAMES) or len(sign_patterns) != len(PROGRAMME_NAMES):
        raise CohortError("every programme needs a block weight and a sign pattern")
    return tuple(
        ProgrammeTarget(
            programme_id=programme_id,
            name=PROGRAMME_NAMES[programme_id],
            conditional_retention=CONDITIONAL_RETENTION[programme_id],
            block_weight=weights[position],
            expected_sign_pattern=sign_patterns[position],
        )
        for position, programme_id in enumerate(PROGRAMME_NAMES)
    )


def retained_programmes() -> tuple[str, ...]:
    return tuple(
        programme_id for programme_id, state in CONDITIONAL_RETENTION.items() if state == RETAINS
    )


def oxygenation_redundant_programmes() -> tuple[str, ...]:
    """Programmes the conditional test is expected to find redundant.

    Ref: Discussion Sec. 4.1 -- one of the six is fully redundant, and it is the
    hypoxia-linked one; the claim under test is programme-specificity rather than
    wholesale independence from oxygenation.
    """
    return tuple(
        programme_id for programme_id, state in CONDITIONAL_RETENTION.items() if state == LOSES
    )


def inconclusive_programmes() -> tuple[str, ...]:
    return tuple(
        programme_id
        for programme_id, state in CONDITIONAL_RETENTION.items()
        if state == INCONCLUSIVE
    )


def conditional_verdicts(
    incremental_gain: Mapping[str, float], null_band: float
) -> Mapping[str, str]:
    """Turn measured incremental gains into the three-way verdict.

    Ref: Results Sec. 2.1 and Discussion Sec. 4.1 -- the three outcomes the study distinguishes are
    a programme that keeps incremental value, one that carries none, and one that is too small to
    call. The rule applied here is therefore: a gain above the null band retains value; a gain at or
    below zero carries none, which is what "the hypoxia-linked programme loses all conditional
    value" means; and a gain inside the band on the positive side is too small to separate from
    nothing and is reported as inconclusive rather than read as a small success.
    """
    verdicts: dict[str, str] = {}
    for programme_id in PROGRAMME_NAMES:
        if programme_id not in incremental_gain:
            raise CohortError(f"no incremental gain was measured for {programme_id}")
        gain = float(incremental_gain[programme_id])
        if gain > null_band:
            verdicts[programme_id] = RETAINS
        elif gain <= 0.0:
            verdicts[programme_id] = LOSES
        else:
            verdicts[programme_id] = INCONCLUSIVE
    return verdicts


def programme_specificity_holds(verdicts: Mapping[str, str]) -> bool:
    """Whether the measured verdicts reproduce the programme-specific reading.

    Ref: Results Sec. 2.1 -- the whole programme is at least partly self-sufficient with
    respect to oxygenation, and the hypoxia-linked block is the one that is not.
    """
    for programme_id in retained_programmes():
        if verdicts.get(programme_id) != RETAINS:
            return False
    for programme_id in oxygenation_redundant_programmes():
        if verdicts.get(programme_id) != LOSES:
            return False
    return True


def unconstrained_substitution_reading(verdicts: Mapping[str, str]) -> str:
    """The reading to report if the proxy had absorbed the programme instead.

    Ref: Results Sec. 2.1 -- the direction of the insignificant residual is reported as
    the genuine outcome of a test that could have gone the other way.
    """
    absorbed = [key for key, value in verdicts.items() if value == LOSES]
    if len(absorbed) >= len(PROGRAMME_NAMES) // 2:
        return "the programme is largely a re-description of oxygenation"
    if absorbed:
        return "the programme is partly, but not wholly, a re-description of oxygenation"
    return "the programme is an axis separate from oxygenation"
