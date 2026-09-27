"""Assay design, clonogenic survival, flux and adjudication."""

from __future__ import annotations

import numpy as np
import pytest

from prism.errors import ConfigError, ProtocolError
from prism.perturbation.clonogenic import (
    fit_linear_quadratic,
    plating_efficiency,
    ratio_by_dose,
    read_assay,
    replicate_spread,
    sensitisation_ratio,
    shoulder_shift,
    surviving_fraction,
)
from prism.perturbation.design import (
    IrradiationScheme,
    LineEligibility,
    default_scheme,
    dose_response_design,
    linear_quadratic_survival,
    protocol_gate,
    screen_lines,
    survival_curve,
)
from prism.perturbation.falsify import (
    FALSIFIED,
    INCONCLUSIVE,
    SUPPORTED,
    PreDeclaredBand,
    ProtocolRecord,
    adjudicate,
    count_failures,
    falsifiability_statement,
    line_substitution_requested,
    mechanistic_consistency,
)
from prism.perturbation.flux import (
    direction_agreement,
    flux_by_metabolite,
    flux_ratio,
    isotope_fraction,
    pool_sufficiency,
    read_flux,
)


def test_a_single_dose_schedule_is_refused() -> None:
    with pytest.raises(ProtocolError):
        IrradiationScheme(
            doses=(2.0, 8.0), fractions_per_dose=1, pre_exposure_hours=24.0, during_exposure=True
        ).validate()
    with pytest.raises(ProtocolError):
        IrradiationScheme(
            doses=(1.0, 2.0, 4.0),
            fractions_per_dose=1,
            pre_exposure_hours=0.0,
            during_exposure=False,
        ).validate()
    scheme = default_scheme()
    assert scheme.fractions_per_dose == 2
    assert scheme.total_dose(0) == pytest.approx(2.0)
    assert scheme.fraction_size(0) == pytest.approx(0.5)
    assert scheme.as_dict()["during_exposure"] is True


def test_the_survival_curve_follows_the_linear_quadratic_model() -> None:
    scheme = default_scheme()
    curve = survival_curve(scheme, 0.20, 0.02)
    assert len(curve) == len(scheme.doses)
    doses = [dose for dose, _ in curve]
    values = [value for _, value in curve]
    assert doses == sorted(doses)
    assert all(b <= a for a, b in zip(values, values[1:], strict=False))
    assert linear_quadratic_survival(0.0, 0.2, 0.02) == pytest.approx(1.0)
    with pytest.raises(ConfigError):
        linear_quadratic_survival(-1.0, 0.2, 0.02)


def test_the_protocol_cannot_be_frozen_after_the_model_review() -> None:
    protocol_gate(model_outputs_examined=False, protocol_frozen=True)
    with pytest.raises(ProtocolError):
        protocol_gate(model_outputs_examined=True, protocol_frozen=True)
    with pytest.raises(ProtocolError):
        protocol_gate(model_outputs_examined=False, protocol_frozen=False)


def test_only_lines_that_can_take_part_are_admitted() -> None:
    candidates = (
        LineEligibility("HCT116", True, True, 0.4),
        LineEligibility("SW480", True, False, 0.0),
        LineEligibility("patient-derived", False, True, 0.3),
    )
    admitted = screen_lines(candidates)
    assert [line.line for line in admitted] == ["HCT116"]
    with pytest.raises(ProtocolError):
        screen_lines(())
    plan = dose_response_design(default_scheme(), ("HCT116",), replicates=3)
    assert len(plan) == len(default_scheme().doses)
    assert plan[0]["arms"] == ("vehicle", "perturbation")


def test_surviving_fraction_and_sensitisation_ratio() -> None:
    colonies = np.asarray([60.0, 55.0])
    plated = np.asarray([6000.0, 6000.0])
    efficiency = plating_efficiency(colonies, plated)
    assert efficiency[0] == pytest.approx(0.01)
    survival = surviving_fraction(colonies / 2, colonies, plated)
    assert 0.0 < survival < 1.0
    assert sensitisation_ratio(0.4, 0.8) == pytest.approx(0.5)
    with pytest.raises(ConfigError):
        plating_efficiency(colonies, np.zeros(2))


def test_the_ratio_is_read_at_every_dose_level() -> None:
    scheme = default_scheme()
    vehicle = {dose: float(np.exp(-(0.2 * dose + 0.02 * dose * dose))) for dose in scheme.doses}
    treated = {dose: value * 0.7 for dose, value in vehicle.items()}
    pairs = ratio_by_dose(treated, vehicle)
    assert len(pairs) == len(scheme.doses)
    assert all(ratio == pytest.approx(0.7) for _, ratio in pairs)
    reading = read_assay("HCT116", treated, vehicle)
    assert reading.radio_sensitised()
    assert reading.ratio == pytest.approx(0.7)
    assert reading.alpha == pytest.approx(0.2, abs=0.02)
    assert (
        shoulder_shift((reading.alpha, reading.beta), (reading.alpha * 1.1, reading.beta))[
            "dominant_component"
        ]
        == "slope"
    )
    mean, spread = replicate_spread([0.68, 0.71, 0.70])
    assert mean == pytest.approx(0.6967, abs=1e-3)
    assert spread > 0.0
    with pytest.raises(ProtocolError):
        fit_linear_quadratic({1.0: 0.9, 2.0: 0.8})


def test_pre_declared_bands_must_match_their_sign() -> None:
    with pytest.raises(ProtocolError):
        PreDeclaredBand("P1", 1, 0.5, 1.4).validate()
    with pytest.raises(ProtocolError):
        PreDeclaredBand("P1", -1, 0.5, 1.4).validate()
    band = PreDeclaredBand("P1", 1, 1.05, 1.4)
    band.validate()
    assert band.contains(1.2)
    assert not band.contains(0.9)
    with pytest.raises(ProtocolError):
        PreDeclaredBand("P1", 0, 0.9, 1.1).validate()


def test_adjudication_returns_the_three_outcomes() -> None:
    scheme = default_scheme()
    vehicle = {dose: float(np.exp(-(0.2 * dose + 0.02 * dose * dose))) for dose in scheme.doses}
    in_band = read_assay("HCT116", {d: v * 0.85 for d, v in vehicle.items()}, vehicle)
    band_positive = PreDeclaredBand("P1", -1, 0.6, 0.97)
    band_increase = PreDeclaredBand("P2", 1, 1.02, 1.4)
    assert adjudicate(band_positive, in_band, 0.003).outcome == SUPPORTED
    assert adjudicate(band_increase, in_band, 0.003).outcome == FALSIFIED
    small = read_assay("HCT116", {d: v * 1.005 for d, v in vehicle.items()}, vehicle)
    assert adjudicate(band_increase, small, 0.003).outcome == INCONCLUSIVE


def test_a_line_outside_the_protocol_is_detected() -> None:
    bands = tuple(PreDeclaredBand(f"P{index}", 1, 1.02, 1.4) for index in range(1, 7))
    protocol = ProtocolRecord(bands=bands, lines=("HCT116",), frozen_before_model_review=True)
    protocol.validate()
    assert protocol.band_for("P1").programme_id == "P1"
    scheme = default_scheme()
    vehicle = {dose: float(np.exp(-(0.2 * dose + 0.02 * dose * dose))) for dose in scheme.doses}
    treated = {dose: value * 0.8 for dose, value in vehicle.items()}
    reading = read_assay("HCT116", treated, vehicle)
    verdicts = tuple(adjudicate(band, reading, 0.003) for band in bands)
    assert not line_substitution_requested(verdicts, protocol)
    counts = count_failures(verdicts)
    assert counts["n_programmes"] == 6.0
    assert counts["n_falsified"] == 6.0
    assert falsifiability_statement(protocol)["falsifiable"] == 1.0
    with pytest.raises(ProtocolError):
        ProtocolRecord(bands=bands, lines=(), frozen_before_model_review=True).validate()
    with pytest.raises(ProtocolError):
        ProtocolRecord(bands=bands, lines=("HCT116",), frozen_before_model_review=False).validate()


def test_the_flux_arm_reads_direction_against_the_emitted_sign() -> None:
    vehicle = {"lactate": (2.0, 40.0), "palmitate": (1.0, 20.0)}
    treated = {"lactate": (4.0, 40.0), "palmitate": (0.4, 20.0)}
    ratios = flux_by_metabolite(treated, vehicle)
    assert ratios["lactate"] == pytest.approx(2.0)
    assert ratios["palmitate"] == pytest.approx(0.4)
    readings = read_flux(ratios, {"lactate": 1, "palmitate": -1})
    agreement = direction_agreement(readings, 0.05)
    assert agreement["agreement_fraction"] == 1.0
    assert float(isotope_fraction(np.array([5.0]), np.array([10.0]))[0]) == pytest.approx(0.5)
    assert flux_ratio(0.5, 0.25) == pytest.approx(2.0)
    assert pool_sufficiency(np.array([10.0, 20.0]), 5.0)
    assert not pool_sufficiency(np.array([10.0, 1.0]), 5.0)
    with pytest.raises(ConfigError):
        isotope_fraction(np.array([20.0]), np.array([10.0]))
    with pytest.raises(ProtocolError):
        read_flux({"lactate": 1.0}, {})
    scheme = default_scheme()
    vehicle_curve = {d: float(np.exp(-(0.2 * d + 0.02 * d * d))) for d in scheme.doses}
    treated_curve = {d: v * 0.7 for d, v in vehicle_curve.items()}
    reading = read_assay("HCT116", treated_curve, vehicle_curve)
    bands = tuple(PreDeclaredBand(f"P{i}", -1, 0.6, 0.97) for i in range(1, 7))
    verdicts = tuple(adjudicate(b, reading, 0.003) for b in bands)
    flux = {"P1": read_flux({"lactate": 2.0}, {"lactate": 1})}
    consistency = mechanistic_consistency(verdicts, flux, 0.05)
    assert consistency["n_with_flux"] == 1.0
    assert consistency["flux_agreement_fraction"] == 1.0
