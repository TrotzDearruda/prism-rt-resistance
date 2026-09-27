"""Multiplicity control over the pre-specified family.

Ref: Methods Sec. 6.10 -- the Holm correction is applied inside one family of five
pre-declared contrasts, no p-value outside that family is corrected, and every comparison
that is not in the family is labelled exploratory rather than family-adjusted. Both halves of
that sentence are load-bearing, so the correction here is explicit about the family it was
handed and refuses to accept a contrast that never entered one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prism.errors import ConfigError

FAMILY_SIZE = 5


@dataclass(frozen=True)
class StepDownEntry:
    """One contrast inside the family, with its raw and adjusted p-value."""

    label: str
    raw_p: float
    adjusted_p: float
    rank: int
    rejected: bool

    def as_dict(self) -> Mapping[str, float | str | bool]:
        return {
            "contrast": self.label,
            "raw_p": self.raw_p,
            "holm_adjusted_p": self.adjusted_p,
            "rank": float(self.rank),
            "rejected": self.rejected,
        }


@dataclass(frozen=True)
class StepDownResult:
    """The family after the correction, in the order the contrasts were supplied."""

    entries: tuple[StepDownEntry, ...]
    alpha: float

    def rejected_labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in self.entries if entry.rejected)

    def adjusted_for(self, label: str) -> float:
        for entry in self.entries:
            if entry.label == label:
                return entry.adjusted_p
        raise ConfigError(f"no contrast named {label!r} is in this family")

    def as_table(self) -> tuple[Mapping[str, float | str | bool], ...]:
        return tuple(entry.as_dict() for entry in self.entries)


def holm_step_down(
    raw_p: Sequence[float], labels: Sequence[str], alpha: float = 0.05
) -> StepDownResult:
    """Holm step-down correction.

    The adjusted value of a contrast is the running maximum of the scaled raw values once the
    family is ordered by raw value, which enforces monotonicity in the ordering; the contrast
    that first fails to clear the scaled level stops the rejections for everything after it.
    """
    if not raw_p:
        raise ConfigError("a family needs at least one p-value")
    if len(raw_p) != len(labels):
        raise ConfigError("every p-value in a family needs a label")
    if len(set(labels)) != len(labels):
        raise ConfigError("contrast labels must be distinct inside a family")
    if not 0.0 < alpha < 1.0:
        raise ConfigError("a family-level alpha must lie strictly inside (0, 1)")
    for value in raw_p:
        if not 0.0 <= value <= 1.0:
            raise ConfigError("a p-value must lie in [0, 1]")

    order = sorted(range(len(raw_p)), key=lambda position: (raw_p[position], labels[position]))
    m = len(raw_p)
    adjusted = [1.0] * m
    running = 0.0
    for rank, position in enumerate(order):
        scaled = min(1.0, float(raw_p[position]) * (m - rank))
        running = max(running, scaled)
        adjusted[position] = running

    stop = m
    for rank, position in enumerate(order):
        level = alpha / float(m - rank)
        if float(raw_p[position]) > level:
            stop = rank
            break
    rejected_positions = {order[rank] for rank in range(stop)}

    entries = tuple(
        StepDownEntry(
            label=labels[position],
            raw_p=float(raw_p[position]),
            adjusted_p=adjusted[position],
            rank=order.index(position),
            rejected=position in rejected_positions,
        )
        for position in range(m)
    )
    return StepDownResult(entries=entries, alpha=alpha)


def family_size_matches(contrasts: Sequence[str], expected: int = FAMILY_SIZE) -> bool:
    """Whether the declared family is the size the analysis plan fixed.

    Ref: Methods Sec. 6.10 -- the family size is a pre-specified quantity, so a run that
    declares a different number of contrasts is reported rather than silently corrected.
    """
    return len(contrasts) == expected and len(set(contrasts)) == len(contrasts)


def exploratory_label(labels: Sequence[str], family: Sequence[str]) -> tuple[str, ...]:
    """Comparisons outside the family, which are reported as exploratory and uncorrected."""
    inside = set(family)
    return tuple(label for label in labels if label not in inside)


def bonferroni_check(raw_p: Sequence[float]) -> tuple[float, ...]:
    """Uniform correction reported beside the step-down so the family cost is visible.

    The step-down is never worse than this and is equal to it when the ordered p-values are
    already monotone after scaling, so the two are reported together.
    """
    if not raw_p:
        raise ConfigError("a family needs at least one p-value")
    m = float(len(raw_p))
    return tuple(min(1.0, float(value) * m) for value in raw_p)
