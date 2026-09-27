"""Adjudication of the perturbation assay.

Ref: Methods Sec. 6.7 -- before any assay is run the anticipated sign and size of each programme
effect are declared, and the chosen line is not replaced when it fails: a line that does not
satisfy the programme beyond the limit is counted as a failed prediction rather than swapped for
a more convenient line. The adjudication below therefore returns three outcomes and never
adjusts the band after seeing the data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prism.errors import ConfigError, ProtocolError
from prism.perturbation.clonogenic import SensitisationReading
from prism.perturbation.flux import FluxReading

SUPPORTED = "supported"
FALSIFIED = "falsified"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class PreDeclaredBand:
    """The sign and magnitude the model predicts for one programme, fixed before the assay."""

    programme_id: str
    anticipated_sign: int
    lower: float
    upper: float

    def validate(self) -> None:
        if self.anticipated_sign == 0:
            raise ProtocolError(
                f"{self.programme_id} carries no sign, so no direction can be falsified"
            )
        if not self.lower < self.upper:
            raise ProtocolError(f"{self.programme_id} carries an empty band")
        if self.anticipated_sign > 0 and self.lower < 1.0:
            raise ProtocolError(
                f"{self.programme_id} predicts an increase but its band admits a decrease"
            )
        if self.anticipated_sign < 0 and self.upper > 1.0:
            raise ProtocolError(
                f"{self.programme_id} predicts a decrease but its band admits an increase"
            )

    def contains(self, ratio: float) -> bool:
        self.validate()
        return self.lower <= float(ratio) <= self.upper

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "programme": self.programme_id,
            "anticipated_sign": self.anticipated_sign,
            "band": [self.lower, self.upper],
        }


@dataclass(frozen=True)
class AssayVerdict:
    """What one programme's assay did."""

    programme_id: str
    line: str
    ratio: float
    outcome: str
    direction_agrees: bool
    in_band: bool

    def as_dict(self) -> Mapping[str, object]:
        return {
            "programme": self.programme_id,
            "line": self.line,
            "ratio": self.ratio,
            "outcome": self.outcome,
            "direction_agrees": self.direction_agrees,
            "in_band": self.in_band,
        }


def adjudicate(
    band: PreDeclaredBand,
    reading: SensitisationReading,
    null_band: float,
) -> AssayVerdict:
    """Read one line's primary output against the pre-declared band."""
    band.validate()
    ratio = float(reading.ratio)
    if band.anticipated_sign > 0:
        direction = ratio > 1.0 + null_band
    else:
        direction = ratio < 1.0 - null_band
    in_band = band.contains(ratio)
    if in_band:
        outcome = SUPPORTED
    elif direction:
        outcome = INCONCLUSIVE
    else:
        outcome = FALSIFIED
    return AssayVerdict(
        programme_id=band.programme_id,
        line=reading.line,
        ratio=ratio,
        outcome=outcome,
        direction_agrees=direction,
        in_band=in_band,
    )


@dataclass(frozen=True)
class ProtocolRecord:
    """The frozen protocol: bands, lines and the order in which things happened."""

    bands: tuple[PreDeclaredBand, ...]
    lines: tuple[str, ...]
    frozen_before_model_review: bool

    def validate(self) -> None:
        if not self.bands:
            raise ProtocolError("a protocol needs at least one pre-declared band")
        if not self.lines:
            raise ProtocolError("a protocol needs at least one admitted line")
        programme_ids = [band.programme_id for band in self.bands]
        if len(set(programme_ids)) != len(programme_ids):
            raise ProtocolError("each programme carries one band")
        for band in self.bands:
            band.validate()
        if not self.frozen_before_model_review:
            raise ProtocolError(
                "the protocol must be frozen before the model output is reviewed, otherwise the "
                "bands are not pre-declared"
            )

    def band_for(self, programme_id: str) -> PreDeclaredBand:
        self.validate()
        for band in self.bands:
            if band.programme_id == programme_id:
                return band
        raise ProtocolError(f"no band was declared for {programme_id}")

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "bands": [band.as_dict() for band in self.bands],
            "lines": list(self.lines),
            "frozen_before_model_review": self.frozen_before_model_review,
        }


def line_substitution_requested(results: Sequence[AssayVerdict], protocol: ProtocolRecord) -> bool:
    """Whether an adjudication used a line outside the frozen protocol.

    Ref: Methods Sec. 6.7 -- a line that fails the programme is counted rather than replaced, so
    a verdict naming a line that was not admitted is a protocol violation and is reported.
    """
    protocol.validate()
    admitted = set(protocol.lines)
    return any(verdict.line not in admitted for verdict in results)


def count_failures(results: Sequence[AssayVerdict]) -> Mapping[str, float]:
    """How many programme predictions survived, failed, or could not be read."""
    if not results:
        raise ProtocolError("no verdict was produced")
    counts = {SUPPORTED: 0, FALSIFIED: 0, INCONCLUSIVE: 0}
    for verdict in results:
        if verdict.outcome not in counts:
            raise ConfigError(f"unknown assay outcome {verdict.outcome!r}")
        counts[verdict.outcome] += 1
    total = float(len(results))
    return {
        "n_programmes": total,
        "n_supported": float(counts[SUPPORTED]),
        "n_falsified": float(counts[FALSIFIED]),
        "n_inconclusive": float(counts[INCONCLUSIVE]),
        "supported_fraction": counts[SUPPORTED] / total,
    }


def mechanistic_consistency(
    verdicts: Sequence[AssayVerdict],
    flux_readings: Mapping[str, Sequence[FluxReading]],
    tolerance: float,
) -> Mapping[str, float]:
    """Whether the mechanistic read-out moved the same way as the primary read-out.

    Ref: Methods Sec. 6.7 -- the flux arm exists to test whether the process advances as
    anticipated rather than merely correlating with the outcome, so a programme whose survival
    ratio moved but whose flux did not is reported as a primary-only effect.
    """
    if not verdicts:
        raise ProtocolError("a consistency reading needs at least one verdict")
    agreeing = 0
    observed = 0
    for verdict in verdicts:
        readings = flux_readings.get(verdict.programme_id)
        if not readings:
            continue
        observed += 1
        if any(reading.direction_agrees(tolerance) for reading in readings):
            agreeing += 1
    return {
        "n_with_flux": float(observed),
        "n_flux_agreeing": float(agreeing),
        "flux_agreement_fraction": (agreeing / float(observed)) if observed else 0.0,
    }


def falsifiability_statement(protocol: ProtocolRecord) -> Mapping[str, float]:
    """Whether every claimed programme could have been contradicted by the design."""
    protocol.validate()
    signed = sum(1 for band in protocol.bands if band.anticipated_sign != 0)
    return {
        "n_bands": float(len(protocol.bands)),
        "n_signed": float(signed),
        "n_lines": float(len(protocol.lines)),
        "falsifiable": float(signed == len(protocol.bands)),
    }
