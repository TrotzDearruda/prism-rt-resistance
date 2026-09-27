"""Discrimination, calibration, decisions, attribution and the interval columns."""

from __future__ import annotations

import numpy as np
import pytest

from prism.errors import ConfigError
from prism.measurement.attribution import (
    attribution_row,
    chance_level,
    constraint_reading,
    fit_classifier,
)
from prism.measurement.calibration import (
    calibration_from_probabilities,
    calibration_report,
    expected_calibration_error,
    logit,
    reliability_curve,
    sigmoid,
    slope_gap_from_one,
)
from prism.measurement.decision import (
    clinical_usefulness_margin,
    decision_report,
    net_benefit,
    optimal_threshold_band,
)
from prism.measurement.discrimination import (
    average_precision,
    discrimination,
    low_resource_report,
    pooled_scores,
    rank_auc,
    standard_error_from_ranks,
    trapezoidal_auc,
)
from prism.measurement.uncertainty import (
    ColumnVariability,
    EnsembleContrast,
    aggregation_reading,
    half_width,
    half_width_table,
)
from prism.numerics.operating_point import (
    operating_point,
    prevalence_sensitivity,
    rule_in_at_deployment,
    rule_in_profile,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)


def make_scores(n_rows: int = 300, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_rows) < 0.3).astype(int)
    scores = labels * 1.4 + rng.normal(0.0, 1.0, size=n_rows)
    return scores, labels


def test_rank_and_trapezoid_readings_agree() -> None:
    scores, labels = make_scores()
    assert rank_auc(scores, labels) == pytest.approx(trapezoidal_auc(scores, labels), abs=1e-9)
    report = discrimination(scores, labels)
    assert report.n_rows == 300
    assert 0.0 <= report.auprc <= 1.0
    assert report.prevalence == pytest.approx(labels.mean())
    assert report.exceeds(0.5)


def test_discrimination_of_a_perfect_and_a_reversed_score() -> None:
    labels = np.asarray([0, 0, 1, 1])
    assert rank_auc(np.asarray([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert rank_auc(np.asarray([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    reversed_precision = average_precision(np.asarray([0.9, 0.8, 0.2, 0.1]), labels)
    assert 0.0 < reversed_precision < 0.5


def test_a_single_class_cannot_be_scored() -> None:
    with pytest.raises(ConfigError):
        rank_auc(np.asarray([0.1, 0.2]), np.asarray([1, 1]))


def test_group_and_stratum_readings_skip_impossible_cells() -> None:
    scores, labels = make_scores(seed=11)
    groups = np.asarray([f"g{index % 3}" for index in range(scores.size)], dtype=object)
    pooled = pooled_scores(scores, labels, groups)
    assert pooled
    by_stratum = low_resource_report(
        {"low": scores[:100], "high": scores[100:]},
        {"low": labels[:100], "high": labels[100:]},
    )
    assert set(by_stratum) == {"low", "high"}
    with pytest.raises(ConfigError):
        standard_error_from_ranks(np.asarray([0.5]), np.asarray([1]))


def test_calibration_slope_recovers_a_known_logistic_scale() -> None:
    rng = np.random.default_rng(17)
    latent = rng.normal(0.0, 1.0, size=2000)
    probabilities = sigmoid(latent)
    labels = (rng.random(2000) < probabilities).astype(int)
    report = calibration_report(logit(probabilities), labels)
    assert report.slope == pytest.approx(1.0, abs=0.25)
    assert report.over_confident() is False or report.slope < 1.0
    assert 0.0 < report.brier < 0.25
    assert slope_gap_from_one(report) == pytest.approx(1.0 - report.slope)
    predicted, observed, counts = reliability_curve(probabilities, labels, bins=8)
    assert predicted.shape == observed.shape == counts.shape
    assert expected_calibration_error(probabilities, labels, bins=8) >= 0.0


def test_a_constant_forecast_is_not_sloped() -> None:
    with pytest.raises(ConfigError):
        calibration_from_probabilities(np.full(5, 0.5), np.zeros(5, dtype=int))


def test_net_benefit_matches_a_hand_computation() -> None:
    probabilities = np.asarray([0.9, 0.8, 0.2, 0.1])
    labels = np.asarray([1, 0, 1, 0])
    threshold = 0.5
    value = net_benefit(probabilities, labels, threshold)
    expected = 1 / 4 - (1 / 4) * (threshold / (1 - threshold))
    assert value == pytest.approx(expected)
    report = decision_report(probabilities, labels, (0.25, 0.5, 0.75))
    assert report.band == (0.25, 0.5, 0.75)
    assert report.best_threshold() in report.band
    margins = clinical_usefulness_margin(report, report)
    assert margins["mean_margin"] == pytest.approx(0.0)
    band = optimal_threshold_band(0.194)
    assert len(band) == 5
    assert band[0] < 0.194 < band[-1]


def test_operating_point_and_rule_in_value() -> None:
    scores, labels = make_scores(seed=23)
    sensitivity, specificity = operating_point(scores, labels, 0.8)
    assert 0.0 <= sensitivity <= 1.0
    assert 0.0 <= specificity <= 1.0
    assert sensitivity_at_specificity(scores, labels, 0.5) > sensitivity_at_specificity(
        scores, labels, 0.9
    )
    assert specificity_at_sensitivity(scores, labels, 0.5) > specificity_at_sensitivity(
        scores, labels, 0.9
    )
    profile = rule_in_profile(scores, labels, 0.194, target_specificity=0.8)
    assert profile.ppv() < 1.0
    at_deployment = rule_in_at_deployment(scores, labels, 0.194)
    assert at_deployment.prevalence == pytest.approx(0.194)
    doubled = prevalence_sensitivity(profile, 0.388)
    assert doubled.ppv() > profile.ppv()


def test_the_attribution_probe_separates_a_leaky_representation() -> None:
    rng = np.random.default_rng(29)
    n_rows = 240
    sites = np.asarray([f"S{index % 4}" for index in range(n_rows)], dtype=object)
    regions = np.asarray([f"R{index % 4}" for index in range(n_rows)], dtype=object)
    offsets = np.asarray([[float(index % 4)] * 5 for index in range(n_rows)])
    leaky = rng.normal(size=(n_rows, 5)) + offsets * 2.5
    constrained = rng.normal(size=(n_rows, 5)) + offsets * 0.04
    leaky_row = attribution_row("unconstrained", leaky, sites, regions, 0.90)
    constrained_row = attribution_row("constrained", constrained, sites, regions, 0.89)
    assert leaky_row.site_accuracy > constrained_row.site_accuracy
    assert constrained_row.site_excess() < 0.2
    assert chance_level(sites) == pytest.approx(0.25)
    reading = constraint_reading((leaky_row, constrained_row))
    assert reading["site_accuracy_drop"] > 0.0
    classifier = fit_classifier(leaky, np.asarray([int(s[1:]) for s in sites]), 4)
    assert classifier.weights.shape == (5, 4)


def test_half_widths_are_reported_per_column() -> None:
    columns = {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.5, 0.5, 0.5, 0.5]}
    rows = half_width_table(columns)
    assert all(isinstance(row, ColumnVariability) for row in rows)
    assert rows[1].half_width == pytest.approx(0.0)
    assert half_width([0.1, 0.2, 0.3, 0.4]) > 0.0
    with pytest.raises(ConfigError):
        half_width([])


def test_the_ensemble_reading_follows_the_table() -> None:
    single = EnsembleContrast("single", 0.857, 0.90, 0.041, 0.884)
    ten = EnsembleContrast("ten", 0.862, 0.95, 0.031, 0.899)
    regime = EnsembleContrast("regime", 0.873, 0.93, 0.036, 0.891)
    assert ten.width_reduction(single) > 0.0
    assert regime.auc_gain(ten) > 0.0
    assert regime.calibration_slope < ten.calibration_slope
    reading = aggregation_reading((single, regime))
    assert reading["width_margin"] > 0.0
    assert reading["coverage_margin"] > 0.0
