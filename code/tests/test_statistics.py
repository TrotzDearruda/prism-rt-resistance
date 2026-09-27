"""Paired comparison, multiplicity control, equivalence and the null band."""

from __future__ import annotations

import numpy as np
import pytest

from prism.errors import ConfigError
from prism.statistics.ci import (
    interval_width_statistics,
    normal_interval,
    patient_cluster_bootstrap,
    reference_inside,
    seed_spread,
)
from prism.statistics.delong import (
    midrank,
    paired_delong,
    pooled_auc_variance,
    structural_components,
    unpaired_contrast,
)
from prism.statistics.equivalence import (
    SuccessCriterion,
    compare_with_reference,
    equivalence_interval,
    per_patient_difference,
    per_patient_standard_error,
    reachability_check,
    two_one_sided,
)
from prism.statistics.family import (
    FAMILY,
    conditional_value_reading,
    ordered_p_values,
    spec_for,
    synergy_statistic,
    validate_family,
)
from prism.statistics.holm import (
    bonferroni_check,
    exploratory_label,
    family_size_matches,
    holm_step_down,
)
from prism.statistics.nullband import (
    NullBand,
    NullBandLedger,
    band_from_config,
    coverage_failure,
    null_on_discrimination,
    read_ablation,
)


def make_pair(n_rows: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_rows) < 0.35).astype(int)
    strong = labels * 1.3 + rng.normal(0.0, 1.0, size=n_rows)
    weak = labels * 0.5 + rng.normal(0.0, 1.0, size=n_rows)
    return strong, weak, labels


def test_the_paired_contrast_matches_hand_computed_areas() -> None:
    strong, weak, labels = make_pair()
    comparison = paired_delong(strong, weak, labels, label="strong-weak")
    positive_strong, negative_strong = structural_components(strong, labels)
    positive_weak, negative_weak = structural_components(weak, labels)
    expected_strong = 0.5 * (positive_strong.mean() + negative_strong.mean())
    expected_weak = 0.5 * (positive_weak.mean() + negative_weak.mean())
    assert comparison.auc_a == pytest.approx(float(expected_strong), abs=1e-9)
    assert comparison.auc_b == pytest.approx(float(expected_weak), abs=1e-9)
    assert comparison.difference == pytest.approx(comparison.auc_a - comparison.auc_b, abs=1e-12)
    assert comparison.p_value < 0.05
    lower, upper = comparison.confidence_interval()
    assert lower < comparison.difference < upper
    assert pooled_auc_variance(strong, labels) > 0.0


def test_ranks_average_ties() -> None:
    ranks = midrank(np.asarray([1.0, 2.0, 2.0, 4.0]))
    assert ranks[1] == pytest.approx(2.5)
    assert ranks[2] == pytest.approx(2.5)


def test_an_unpaired_contrast_is_wider_than_a_paired_one() -> None:
    strong, weak, labels = make_pair(seed=5)
    paired = paired_delong(strong, weak, labels)
    unpaired = unpaired_contrast(strong, weak, labels, labels)
    assert unpaired.standard_error >= paired.standard_error * 0.5


def test_holm_matches_a_hand_worked_step_down() -> None:
    raw = [0.001, 0.008, 0.02, 0.04, 0.30]
    labels = [f"c{index}" for index in range(5)]
    result = holm_step_down(raw, labels, alpha=0.05)
    adjusted = [entry.adjusted_p for entry in result.entries]
    assert adjusted == pytest.approx([0.005, 0.032, 0.06, 0.08, 0.30], abs=1e-9)
    assert result.rejected_labels() == ("c0", "c1")
    assert result.adjusted_for("c0") == pytest.approx(0.005)
    uniform = bonferroni_check(raw)
    assert uniform[0] == pytest.approx(0.005)
    assert all(a <= b + 1e-12 for a, b in zip(adjusted, uniform, strict=True))


def test_the_family_is_the_size_the_plan_fixed() -> None:
    validate_family()
    assert len(FAMILY) == 5
    assert family_size_matches([spec.name for spec in FAMILY])
    assert spec_for("primary-equivalence").kind == "equivalence"
    values = ordered_p_values({spec.name: 0.01 for spec in FAMILY})
    assert len(values) == 5
    with pytest.raises(ConfigError):
        ordered_p_values({"primary-equivalence": 0.01})
    assert exploratory_label(("a", "b"), ("a",)) == ("b",)


def test_the_synergy_statistic_reads_super_additivity() -> None:
    reading = synergy_statistic(0.038, 0.069, 0.127)
    assert reading["additive_expectation"] == pytest.approx(0.107)
    assert reading["super_additive_excess"] == pytest.approx(0.02)
    assert reading["interchangeable"] == 0.0
    conditional = conditional_value_reading({"P1": 0.05, "P2": -0.01}, 0.003, ("P1", "P2"))
    assert conditional["n_cleared"] == 1.0


def test_the_equivalence_reading_needs_both_limbs() -> None:
    result = two_one_sided(-0.006, 0.004, 0.015)
    assert result.equivalent
    assert result.margin_used() == pytest.approx(0.4)
    assert result.interval[0] < -0.006 < result.interval[1]
    criterion = SuccessCriterion(-0.006, 0.015, 0.039)
    assert criterion.primary_limb() and criterion.external_limb()
    assert criterion.satisfied()
    assert criterion.margin_slack() == pytest.approx(0.009)
    failing = SuccessCriterion(-0.02, 0.015, -0.01)
    assert not failing.satisfied()
    assert compare_with_reference(0.039, -0.05) == pytest.approx(0.089)
    assert reachability_check(0.015, 0.02)["resolvable"] == 0.0
    assert reachability_check(0.015, 0.004)["resolvable"] == 1.0
    assert equivalence_interval(0.0, 0.01)[0] < 0.0


def test_per_patient_averaging_does_not_let_a_heavy_patient_dominate() -> None:
    scores_a = np.asarray([1.0, 1.0, 1.0, 0.0])
    scores_b = np.asarray([0.0, 0.0, 0.0, 0.0])
    groups = np.asarray(["p1", "p1", "p1", "p2"], dtype=object)
    assert per_patient_difference(scores_a, scores_b, groups) == pytest.approx(0.5)
    assert per_patient_standard_error(scores_a, scores_b, groups) > 0.0


def test_the_null_band_reads_a_movement() -> None:
    band = NullBand()
    assert band.resolves(0.01)
    assert not band.resolves(0.001)
    assert band.direction(0.0) == "null"
    assert band.direction(-0.01) == "decrease"
    movement = read_ablation("minus_alignment", "external_auc", 0.861, 0.823)
    assert movement.resolves
    assert movement.movement == pytest.approx(-0.038)
    assert null_on_discrimination(0.001)
    assert coverage_failure(0.90, 0.80)
    ledger = NullBandLedger(
        band=band,
        movements=(
            movement,
            read_ablation("minus_interval_head", "external_auc", 0.861, 0.860),
        ),
    )
    assert len(ledger.unresolved()) == 1
    assert ledger.columns_all_null() == ()
    assert band_from_config(0.003).value == pytest.approx(0.003)
    with pytest.raises(ConfigError):
        band_from_config(-1.0).validate()


def test_patient_bootstrap_and_seed_spread() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    patients = [f"p{index // 2}" for index in range(len(values))]
    interval = patient_cluster_bootstrap(
        values, patients, lambda array: float(array.mean()), draws=200, seed=7
    )
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.half_width > 0.0
    assert reference_inside(interval, 0.45) is True or not interval.contains(0.45)
    wald = normal_interval(0.5, 0.1)
    assert wald.lower == pytest.approx(0.5 - 1.96 * 0.1, abs=1e-3)
    centre, spread = seed_spread([0.9, 0.91, 0.89])
    assert centre == pytest.approx(0.9, abs=1e-9)
    assert spread > 0.0
    stats = interval_width_statistics([0.1, 0.2, 0.3])
    assert stats["mean_width"] == pytest.approx(0.2)
