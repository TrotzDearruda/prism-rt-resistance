"""Clonogenic survival and the sensitisation ratio.

Ref: Methods Sec. 6.7 -- the primary output of the in-vitro arm is clonogenic survival expressed
as a sensitisation ratio rho_p, measured under the multi-dose schedule so that a change in the
shoulder of the curve is distinguished from a change in its slope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError, ProtocolError


def plating_efficiency(observed_colonies: np.ndarray, plated_cells: np.ndarray) -> np.ndarray:
    """Colonies over cells plated, with a guard against a zero denominator."""
    colonies = np.asarray(observed_colonies, dtype=np.float64)
    plated = np.asarray(plated_cells, dtype=np.float64)
    if colonies.shape != plated.shape:
        raise ConfigError("colony counts and plated cell counts must align")
    if np.any(plated <= 0.0):
        raise ConfigError("a plating efficiency needs a positive number of cells plated")
    if np.any(colonies < 0.0):
        raise ConfigError("a colony count cannot be negative")
    efficiency: np.ndarray = colonies / plated
    return efficiency


def surviving_fraction(
    treated_colonies: np.ndarray,
    control_colonies: np.ndarray,
    plated_cells: np.ndarray,
) -> float:
    """Surviving fraction of a treated arm against its own unirradiated control."""
    treated = plating_efficiency(treated_colonies, plated_cells)
    control = plating_efficiency(control_colonies, plated_cells)
    denominator = float(control.mean())
    if denominator <= 0.0:
        raise ProtocolError("the control arm produced no colonies, so no ratio is defined")
    return float(treated.mean()) / denominator


def sensitisation_ratio(treated_survival: float, vehicle_survival: float) -> float:
    """rho_p: survival under the perturbation relative to survival under the vehicle.

    A value below one means the perturbation sensitised the line to irradiation; a value above
    one means it protected the line. The ratio is the primary output because it is read against
    the effect the model predicted, whereas an absolute surviving fraction also moves with the
    intrinsic radiosensitivity of the line.
    """
    if vehicle_survival <= 0.0 or treated_survival < 0.0:
        raise ConfigError("a sensitisation ratio needs a positive vehicle survival")
    return float(treated_survival) / float(vehicle_survival)


def ratio_by_dose(
    treated: Mapping[float, float], vehicle: Mapping[float, float]
) -> tuple[tuple[float, float], ...]:
    """Sensitisation ratio at each dose level, requiring the two arms to share a schedule."""
    if set(treated) != set(vehicle):
        raise ProtocolError("the treated and vehicle arms must share the dose schedule")
    levels = sorted(treated)
    if len(levels) < 3:
        raise ProtocolError("a single- or two-dose schedule cannot separate repair from resistance")
    return tuple((dose, sensitisation_ratio(treated[dose], vehicle[dose])) for dose in levels)


def fit_linear_quadratic(
    survival_by_dose: Mapping[float, float], *, ridge: float = 1e-3
) -> tuple[float, float]:
    """Least-squares fit of the linear-quadratic coefficients to a survival curve.

    The fit is taken on the log survival against dose, so the shoulder parameter enters as a
    quadratic column; with fewer than three dose levels the two coefficients are not separately
    identifiable and the fit refuses rather than returning a degenerate pair.
    """
    if len(survival_by_dose) < 3:
        raise ProtocolError("a linear-quadratic fit needs at least three dose levels")
    doses = np.asarray(sorted(survival_by_dose), dtype=np.float64)
    survival = np.asarray([survival_by_dose[dose] for dose in doses], dtype=np.float64)
    if np.any(survival <= 0.0) or np.any(survival > 1.0):
        raise ConfigError("a surviving fraction must lie in (0, 1]")
    design = np.column_stack([doses, doses * doses])
    response = -np.log(survival)
    penalty = ridge * np.eye(2, dtype=np.float64)
    gram = design.T @ design + penalty
    solution = np.linalg.solve(gram, design.T @ response)
    return float(max(solution[0], 0.0)), float(max(solution[1], 0.0))


@dataclass(frozen=True)
class SensitisationReading:
    """The primary output of the assay for one line."""

    line: str
    ratio: float
    alpha: float
    beta: float
    ratio_by_level: tuple[float, ...]

    def radio_sensitised(self) -> bool:
        return self.ratio < 1.0

    def as_dict(self) -> Mapping[str, object]:
        return {
            "line": self.line,
            "sensitisation_ratio": self.ratio,
            "alpha": self.alpha,
            "beta": self.beta,
            "ratio_by_dose_level": list(self.ratio_by_level),
            "radio_sensitised": self.radio_sensitised(),
        }


def read_assay(
    line: str,
    treated_by_dose: Mapping[float, float],
    vehicle_by_dose: Mapping[float, float],
) -> SensitisationReading:
    """Assemble the reading from the two arms."""
    pairs = ratio_by_dose(treated_by_dose, vehicle_by_dose)
    ratios = np.asarray([ratio for _, ratio in pairs], dtype=np.float64)
    alpha, beta = fit_linear_quadratic(vehicle_by_dose)
    return SensitisationReading(
        line=line,
        ratio=float(ratios.mean()),
        alpha=alpha,
        beta=beta,
        ratio_by_level=tuple(float(ratio) for ratio in ratios),
    )


def shoulder_shift(
    vehicle: tuple[float, float], treated: tuple[float, float]
) -> Mapping[str, object]:
    """How the fit moved between the two arms.

    Ref: Methods Sec. 6.7 -- a multi-dose schedule is used precisely so that the shoulder and the
    slope can be compared separately; a shift carried by the quadratic coefficient is reported
    differently from one carried by the linear coefficient.
    """
    alpha_shift = treated[0] - vehicle[0]
    beta_shift = treated[1] - vehicle[1]
    dominant = "shoulder" if abs(beta_shift) > abs(alpha_shift) else "slope"
    return {
        "alpha_shift": alpha_shift,
        "beta_shift": beta_shift,
        "dominant_component": dominant,
    }


def replicate_spread(per_replicate: Sequence[float]) -> tuple[float, float]:
    """Mean and spread of a replicate series, reported next to the ratio."""
    values = np.asarray(per_replicate, dtype=np.float64)
    if values.size < 2:
        raise ConfigError("a replicate spread needs at least two replicates")
    return float(values.mean()), float(values.std(ddof=1))
