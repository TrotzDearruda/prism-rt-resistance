"""Stable-isotope-resolved metabolic flux.

Ref: Methods Sec. 6.7 -- the mechanistic secondary output is stable-isotope-resolved metabolic
flux, which assesses whether the process advances as anticipated instead of merely correlating
with the outcome. The quantity that matters is therefore the direction and size of the shift in
labelling, read against the direction the model predicted for the programme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError, ProtocolError


def isotope_fraction(labelled: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Fraction of a metabolite pool carrying the label."""
    heavy = np.asarray(labelled, dtype=np.float64)
    pool = np.asarray(total, dtype=np.float64)
    if heavy.shape != pool.shape:
        raise ConfigError("labelled and total abundances must align")
    if np.any(pool <= 0.0):
        raise ConfigError("a labelling fraction needs a positive pool size")
    if np.any(heavy < 0.0) or np.any(heavy > pool):
        raise ConfigError("labelled abundance cannot exceed the pool it sits in")
    fraction: np.ndarray = heavy / pool
    return fraction


def flux_ratio(treated_fraction: float, vehicle_fraction: float, *, floor: float = 1e-6) -> float:
    """Labelling under the perturbation relative to the vehicle.

    A ratio above one means the perturbation increased flux through the metabolite, which is the
    direction the programme sign predicts for a resistance-driving programme.
    """
    if vehicle_fraction <= 0.0:
        raise ProtocolError("a flux ratio needs a positive vehicle labelling fraction")
    if treated_fraction < 0.0:
        raise ConfigError("a labelling fraction cannot be negative")
    return float(max(treated_fraction, floor)) / float(vehicle_fraction)


def flux_by_metabolite(
    treated: Mapping[str, tuple[float, float]], vehicle: Mapping[str, tuple[float, float]]
) -> Mapping[str, float]:
    """Per-metabolite flux ratio from labelled and total abundances in both arms."""
    if set(treated) != set(vehicle):
        raise ProtocolError("the treated and vehicle arms must share a metabolite panel")
    ratios: dict[str, float] = {}
    for name in sorted(treated):
        heavy_treated, pool_treated = treated[name]
        heavy_vehicle, pool_vehicle = vehicle[name]
        treated_fraction = float(
            isotope_fraction(np.array([heavy_treated]), np.array([pool_treated]))[0]
        )
        vehicle_fraction = float(
            isotope_fraction(np.array([heavy_vehicle]), np.array([pool_vehicle]))[0]
        )
        ratios[name] = flux_ratio(treated_fraction, vehicle_fraction)
    return ratios


@dataclass(frozen=True)
class FluxReading:
    """One metabolite's labelling shift next to the direction the programme predicts."""

    metabolite: str
    ratio: float
    anticipated_sign: int

    def direction_agrees(self, tolerance: float) -> bool:
        if self.anticipated_sign == 0:
            raise ConfigError("a zero anticipated sign states no direction to agree with")
        if self.anticipated_sign > 0:
            return self.ratio > 1.0 + tolerance
        return self.ratio < 1.0 - tolerance

    def as_dict(self) -> Mapping[str, object]:
        return {
            "metabolite": self.metabolite,
            "flux_ratio": self.ratio,
            "anticipated_sign": self.anticipated_sign,
        }


def read_flux(
    ratios: Mapping[str, float], anticipated_signs: Mapping[str, int]
) -> tuple[FluxReading, ...]:
    """Pair a flux panel with the programme signs the model emitted."""
    if not ratios:
        raise ProtocolError("a flux reading needs at least one metabolite")
    missing = [name for name in ratios if name not in anticipated_signs]
    if missing:
        raise ProtocolError(f"the panel carries metabolites the emission does not sign: {missing}")
    return tuple(
        FluxReading(
            metabolite=name,
            ratio=float(ratios[name]),
            anticipated_sign=int(anticipated_signs[name]),
        )
        for name in sorted(ratios)
    )


def direction_agreement(readings: Sequence[FluxReading], tolerance: float) -> Mapping[str, float]:
    """How much of the panel moved the way the programme predicted."""
    if not readings:
        raise ProtocolError("a direction reading needs at least one metabolite")
    agree = sum(1 for reading in readings if reading.direction_agrees(tolerance))
    ratios = np.asarray([reading.ratio for reading in readings], dtype=np.float64)
    return {
        "n_metabolites": float(len(readings)),
        "n_agreeing": float(agree),
        "agreement_fraction": agree / float(len(readings)),
        "median_ratio": float(np.median(ratios)),
        "log2_median_shift": float(np.median(np.log2(np.clip(ratios, 1e-9, None)))),
    }


def pool_sufficiency(total: np.ndarray, minimum: float) -> bool:
    """Whether a labelling measurement had enough pool to be interpreted.

    A fraction computed on a pool below the detection floor is reported as uninterpretable
    rather than as a small ratio.
    """
    pools = np.asarray(total, dtype=np.float64)
    if pools.size == 0:
        raise ConfigError("a sufficiency check needs at least one pool measurement")
    return bool(np.all(pools >= minimum))
