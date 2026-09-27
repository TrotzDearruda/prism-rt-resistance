"""Perturbation-assay design.

Ref: Methods Sec. 6.7 -- the in-vitro arm is falsifiable through a protocol fixed before the
model output is verified. Only cell lines that can be used in vitro take part. A multi-dose
irradiation scheme is used rather than a single dose, because a single-dose survival curve
expresses resistance through sub-lethal damage repair in a way the design cannot separate. The
perturbation is applied before and during irradiation. The primary output is clonogenic survival
expressed as a sensitisation ratio, and the protocol fixes the anticipated sign and size of that
ratio before any assay is run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError, ProtocolError

DOSE_UNIT = "Gy"


@dataclass(frozen=True)
class IrradiationScheme:
    """A multi-dose schedule with the windows in which the perturbation is present."""

    doses: tuple[float, ...]
    fractions_per_dose: int
    pre_exposure_hours: float
    during_exposure: bool

    def validate(self) -> None:
        if len(self.doses) < 3:
            raise ProtocolError(
                "a single-dose or two-dose schedule cannot separate sub-lethal repair from "
                "resistance; at least three dose levels are required"
            )
        if any(dose <= 0.0 for dose in self.doses):
            raise ProtocolError("a dose level must be positive")
        if self.fractions_per_dose < 1:
            raise ProtocolError("a scheme must fractionate at least once per dose level")
        if self.pre_exposure_hours < 0.0:
            raise ProtocolError("a pre-exposure window cannot be negative")
        if not self.during_exposure:
            raise ProtocolError("the perturbation is present during exposure as well as before it")

    def total_dose(self, index: int) -> float:
        self.validate()
        return self.doses[index] * float(self.fractions_per_dose)

    def fraction_size(self, index: int) -> float:
        self.validate()
        return self.doses[index] / float(self.fractions_per_dose)

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "dose_levels": list(self.doses),
            "fractions_per_dose": self.fractions_per_dose,
            "pre_exposure_hours": self.pre_exposure_hours,
            "during_exposure": self.during_exposure,
            "unit": DOSE_UNIT,
        }


def default_scheme(
    doses: Sequence[float] = (1.0, 2.0, 4.0, 6.0, 8.0),
    fractions_per_dose: int = 2,
    pre_exposure_hours: float = 24.0,
) -> IrradiationScheme:
    scheme = IrradiationScheme(
        doses=tuple(float(dose) for dose in doses),
        fractions_per_dose=fractions_per_dose,
        pre_exposure_hours=pre_exposure_hours,
        during_exposure=True,
    )
    scheme.validate()
    return scheme


def linear_quadratic_survival(dose: float, alpha: float, beta: float) -> float:
    """Surviving fraction under the linear-quadratic model."""
    if dose < 0.0:
        raise ConfigError("a dose cannot be negative")
    if alpha < 0.0 or beta < 0.0:
        raise ConfigError("linear-quadratic coefficients cannot be negative")
    return float(np.exp(-(alpha * dose + beta * dose * dose)))


def survival_curve(
    scheme: IrradiationScheme, alpha: float, beta: float
) -> tuple[tuple[float, float], ...]:
    """Surviving fraction at every dose level of a scheme."""
    scheme.validate()
    return tuple((dose, linear_quadratic_survival(dose, alpha, beta)) for dose in scheme.doses)


@dataclass(frozen=True)
class LineEligibility:
    """Whether a line may take part in the assay at all."""

    line: str
    in_vitro_capable: bool
    programme_present: bool
    programme_effect: float

    def eligible(self) -> bool:
        return self.in_vitro_capable and self.programme_present

    def as_dict(self) -> Mapping[str, object]:
        return {
            "line": self.line,
            "in_vitro_capable": self.in_vitro_capable,
            "programme_present": self.programme_present,
            "programme_effect": self.programme_effect,
            "eligible": self.eligible(),
        }


def screen_lines(lines: Sequence[LineEligibility]) -> tuple[LineEligibility, ...]:
    """Keep the lines the protocol admits, in the order they were declared."""
    if not lines:
        raise ProtocolError("a screen needs at least one candidate line")
    names = [line.line for line in lines]
    if len(set(names)) != len(names):
        raise ProtocolError("candidate lines must be distinct")
    return tuple(line for line in lines if line.eligible())


def protocol_gate(model_outputs_examined: bool, protocol_frozen: bool) -> None:
    """Refuse to let the assay begin after the model output has been inspected.

    Ref: Methods Sec. 6.7 -- the anticipated bands are established before the model output is
    verified, which is what makes the assay a test of the programme rather than a description of
    it.
    """
    if not protocol_frozen:
        raise ProtocolError("the assay protocol must be frozen before any assay is run")
    if model_outputs_examined:
        raise ProtocolError(
            "the protocol cannot be frozen after the model outputs were examined; the bands "
            "would no longer be pre-declared"
        )


def dose_response_design(
    scheme: IrradiationScheme, lines: Sequence[str], replicates: int
) -> tuple[Mapping[str, object], ...]:
    """One row per line and dose level, with the arms the protocol requires."""
    scheme.validate()
    if not lines:
        raise ProtocolError("a dose-response design needs at least one line")
    if replicates < 2:
        raise ProtocolError("a dose-response design needs at least two replicates")
    rows: list[Mapping[str, object]] = []
    for line in lines:
        for position, dose in enumerate(scheme.doses):
            rows.append(
                {
                    "line": line,
                    "dose_level": dose,
                    "unit": DOSE_UNIT,
                    "fractions": scheme.fractions_per_dose,
                    "arms": ("vehicle", "perturbation"),
                    "replicates": replicates,
                    "pre_exposure_hours": scheme.pre_exposure_hours,
                    "during_exposure": scheme.during_exposure,
                    "dose_index": position,
                }
            )
    return tuple(rows)
