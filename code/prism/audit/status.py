"""Verification status bookkeeping.

Every claim ends in one of four states and the four are not interchangeable. ``PASS`` means the
claim was executed and held on the data that were available; ``FAIL`` means it was executed and did
not hold; ``NOT_RUN`` means it was not executed, either because the article does not fix the value
it would be checked against or because the design has no execution step; ``BLOCKED`` means the data
the claim needs are not obtainable here, so no execution was possible and none is implied. A run
that executed some claims and not others is reported as ``PARTIALLY_VERIFIED``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from prism.errors import ConfigError

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"

VERIFIED = "VERIFIED"
PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
UNVERIFIED = "UNVERIFIED"

STATUSES: tuple[str, ...] = (PASS, FAIL, NOT_RUN, BLOCKED)

MANUSCRIPT_ARITHMETIC = "manuscript-arithmetic"
CODE_BEHAVIOUR = "code-behaviour"


@dataclass(frozen=True)
class ClaimOutcome:
    """What one verification step found for one claim."""

    claim_id: str
    status: str
    layer: str
    check: str
    detail: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.status not in STATUSES:
            raise ConfigError(f"unknown verification status {self.status!r}")
        if self.layer not in {MANUSCRIPT_ARITHMETIC, CODE_BEHAVIOUR}:
            raise ConfigError(f"unknown verification layer {self.layer!r}")
        if not self.check:
            raise ConfigError(f"claim {self.claim_id} carries an unnamed check")

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "status": self.status,
            "layer": self.layer,
            "check": self.check,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def overall_status(outcomes: Sequence[ClaimOutcome]) -> str:
    """Roll the per-claim outcomes up into one word for the release."""
    if not outcomes:
        return UNVERIFIED
    statuses = {outcome.status for outcome in outcomes}
    if statuses == {PASS}:
        return VERIFIED
    if statuses <= {NOT_RUN, BLOCKED}:
        return UNVERIFIED
    return PARTIALLY_VERIFIED


def layer_status(outcomes: Sequence[ClaimOutcome], layer: str) -> str:
    """One verification layer folded to a single claim status.

    The layer fields of the report speak the claim vocabulary rather than the release vocabulary, so
    that a reader can see at a glance whether the code layer failed somewhere or whether every check
    on it passed, without having to work out what a partial release means for one layer.
    """
    selected = [outcome for outcome in outcomes if outcome.layer == layer]
    if not selected:
        return NOT_RUN
    position = {status: rank for rank, status in enumerate((NOT_RUN, BLOCKED, PASS, FAIL))}
    return max((outcome.status for outcome in selected), key=lambda status: position[status])


def tally(outcomes: Sequence[ClaimOutcome]) -> Mapping[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for outcome in outcomes:
        counts[outcome.status] += 1
    return counts


@dataclass(frozen=True)
class VerificationReport:
    """The whole verification body, ready to be written next to the release."""

    project: str
    outcomes: tuple[ClaimOutcome, ...]
    environment: Mapping[str, object]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> Mapping[str, object]:
        for outcome in self.outcomes:
            outcome.validate()
        code_status = layer_status(self.outcomes, CODE_BEHAVIOUR)
        manuscript_status = layer_status(self.outcomes, MANUSCRIPT_ARITHMETIC)
        return {
            "producer": "prism.audit.status",
            "project": self.project,
            "overall_status": overall_status(self.outcomes),
            "code_status": code_status,
            "manuscript_arithmetic_status": manuscript_status,
            "tally": tally(self.outcomes),
            "environment": dict(self.environment),
            "notes": list(self.notes),
            "claims": {outcome.claim_id: outcome.as_dict() for outcome in self.outcomes},
        }


def write_verification_report(path: Path, report: VerificationReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def require_execution_evidence(outcomes: Sequence[ClaimOutcome]) -> None:
    """Refuse a report in which nothing at all was run.

    A release that reports only ``NOT_RUN`` and ``BLOCKED`` rows is honest but says nothing about
    whether the code works, so the driver calls this before writing its report.
    """
    if not any(outcome.status in {PASS, FAIL} for outcome in outcomes):
        raise ConfigError(
            "no claim was executed, so the release carries no runtime evidence at all"
        )


def failing_claims(outcomes: Sequence[ClaimOutcome]) -> tuple[str, ...]:
    return tuple(outcome.claim_id for outcome in outcomes if outcome.status == FAIL)


def blocked_claims(outcomes: Sequence[ClaimOutcome]) -> tuple[str, ...]:
    return tuple(outcome.claim_id for outcome in outcomes if outcome.status == BLOCKED)
