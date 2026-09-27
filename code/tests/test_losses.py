"""The four terms of Eq. (1) and the constraint of Eq. (2)."""

from __future__ import annotations

import pytest
import torch

from prism.errors import ConfigError, FeasibilityError
from prism.objectives.alignment import (
    AlignmentLoss,
    covariance_matrix,
    gram_matrix,
    regional_alignment,
)
from prism.objectives.feasibility import (
    FeasibilityFloor,
    FeasibilityLedger,
    assert_feasible_at_end,
    evaluate_epoch,
    state_snapshot,
    trade_off_point,
)
from prism.objectives.invariance import (
    AdversarialPenalty,
    AttributionSummary,
    VariancePenalty,
    between_group_covariance,
    flatten_attribution,
    penalty_for_mode,
)
from prism.objectives.outcome import (
    OutcomeLoss,
    balanced_positive_weight,
    brier_decomposition_free,
    implied_prevalence,
    outcome_accuracy,
)
from prism.objectives.programme import (
    GroupPenalty,
    SignPenalty,
    block_activity,
    effective_programmes,
    group_norms,
    sparsity_fraction,
)
from prism.objectives.total import (
    ConstrainedObjective,
    ObjectiveWeights,
    blend,
    effective_floor,
    zero_terms,
)


def test_the_outcome_term_matches_a_hand_computed_mean() -> None:
    loss = OutcomeLoss()
    logits = torch.tensor([2.0, -2.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 0.5])
    value = float(loss(logits, labels))
    expected = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    assert value == pytest.approx(expected, abs=1e-6)
    assert implied_prevalence(torch.sigmoid(logits)) == pytest.approx(
        float(torch.sigmoid(logits).mean())
    )
    assert outcome_accuracy(torch.tensor([1.0, -1.0]), torch.tensor([1, 0])) == 1.0
    assert balanced_positive_weight(torch.tensor([1.0, 0.0, 0.0])) == 2.0


def test_brier_score_of_a_perfect_and_a_useless_forecast() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert brier_decomposition_free(labels, labels) == 0.0
    assert brier_decomposition_free(torch.full((4,), 0.5), labels) == pytest.approx(0.25)


def test_the_matching_half_separates_a_row_permutation_and_stays_non_negative() -> None:
    torch.manual_seed(3)
    metabolite = torch.randn(48, 10)
    loss = AlignmentLoss()
    matched = metabolite * 0.9 + torch.randn(48, 10) * 0.05
    scrambled = metabolite[torch.randperm(48)]
    matched_value = float(loss(metabolite, matched)["matching"])
    scrambled_value = float(loss(metabolite, scrambled)["matching"])
    assert 0.0 <= matched_value < scrambled_value


def test_the_second_moment_half_separates_an_axis_permutation() -> None:
    torch.manual_seed(5)
    metabolite = torch.randn(48, 10)
    loss = AlignmentLoss()
    matched = metabolite * 0.95 + torch.randn(48, 10) * 0.02
    axis_scrambled = metabolite[:, torch.randperm(10)]
    assert float(loss(metabolite, matched)["second_moment"]) < float(
        loss(metabolite, axis_scrambled)["second_moment"]
    )


def test_the_second_moment_half_cannot_see_a_row_permutation() -> None:
    torch.manual_seed(7)
    metabolite = torch.randn(48, 10)
    loss = AlignmentLoss()
    row_scrambled = metabolite[torch.randperm(48)]
    assert float(loss(metabolite, row_scrambled)["second_moment"]) == pytest.approx(
        float(loss(metabolite, metabolite)["second_moment"]), abs=1e-5
    )
    assert covariance_matrix(metabolite).shape == (10, 10)
    assert gram_matrix(metabolite).shape == (48, 48)


def test_a_second_moment_match_needs_enough_rows() -> None:
    with pytest.raises(ConfigError):
        AlignmentLoss()(torch.randn(2, 4), torch.randn(2, 4))


def test_regional_alignment_averages_over_regions() -> None:
    torch.manual_seed(11)
    representation = {"Region I": torch.randn(16, 8), "Region II": torch.randn(16, 8)}
    metabolite = {"Region I": torch.randn(16, 8), "Region II": torch.randn(16, 8)}
    terms = regional_alignment(representation, metabolite, AlignmentLoss())
    assert set(terms) == {"alignment", "matching", "second_moment"}
    assert float(terms["alignment"]) >= 0.0


def test_the_variance_penalty_falls_when_the_site_means_are_removed() -> None:
    torch.manual_seed(13)
    sites = torch.arange(4).repeat_interleave(20)
    offsets = torch.randn(4, 6) * 3.0
    rows = torch.randn(80, 6) + offsets[sites]
    penalty = VariancePenalty()
    before = float(penalty(rows, sites, 4)["invariance"])
    centroids = torch.stack([rows[sites == site].mean(dim=0) for site in range(4)])
    after = float(penalty((rows - centroids[sites]), sites, 4)["invariance"])
    assert after < before
    assert between_group_covariance(rows, sites, 4).shape == (6, 6)
    assert float(between_group_covariance(rows, sites, 4).diagonal().sum()) > 0.0


def test_the_adversarial_penalty_is_a_cross_entropy() -> None:
    logits = torch.randn(12, 5)
    sites = torch.randint(0, 5, (12,))
    value = float(AdversarialPenalty()(logits, sites)["invariance"])
    expected = float(torch.nn.functional.cross_entropy(logits, sites))
    assert value == pytest.approx(expected)
    assert (
        float(
            penalty_for_mode("adversarial", torch.randn(4, 3), sites[:4], 5, logits[:4])[
                "invariance"
            ]
        )
        > 0.0
    )


def test_an_unknown_invariance_reading_is_refused() -> None:
    with pytest.raises(ConfigError):
        penalty_for_mode("mixture", torch.randn(4, 3), torch.arange(4), 4)


def test_the_group_penalty_acts_on_blocks(lexicon) -> None:
    coefficients = torch.randn(6, lexicon.size) * 0.3
    norms = group_norms(coefficients, lexicon.index_blocks(), lexicon.block_weights())
    assert norms.shape == (6, lexicon.n_blocks())
    penalty = float(GroupPenalty()(coefficients, lexicon))
    assert penalty > 0.0
    assert float(GroupPenalty(square=False)(coefficients, lexicon)) > 0.0
    assert 0.0 <= sparsity_fraction(coefficients) < 1.0
    activity = block_activity(coefficients, lexicon)
    assert set(activity) == set(lexicon.programme_ids)
    assert effective_programmes(coefficients, lexicon, 0.0)


def test_the_sign_penalty_charges_a_wrong_direction() -> None:
    expected = (1, -1, 1, -1)
    aligned = torch.tensor([[0.5, -0.5, 0.5, -0.5]])
    against = -aligned
    penalty = SignPenalty()
    assert float(penalty(aligned, expected)) == 0.0
    assert float(penalty(against, expected)) > 0.0


def test_the_objective_combines_the_four_terms(lexicon) -> None:
    objective = ConstrainedObjective.build(ObjectiveWeights())
    coefficients = torch.randn(12, lexicon.size) * 0.2
    logits = torch.randn(12)
    labels = torch.randint(0, 2, (12,)).to(torch.float32)
    representation = torch.randn(12, 16)
    metabolite = torch.randn(12, lexicon.size)
    sites = torch.arange(12) % 4
    result = objective.stage_two(
        logits,
        labels,
        representation,
        coefficients,
        metabolite,
        coefficients,
        lexicon,
        lexicon.expected_signs(),
        sites,
        4,
    )
    for name in ("outcome", "alignment", "invariance", "programme", "group", "sign"):
        assert name in result.terms
    assert float(result.total.detach()) >= 0.0
    stage_one = objective.stage_one(logits, labels, coefficients, lexicon, lexicon.expected_signs())
    assert stage_one.active["alignment"] is False
    assert stage_one.active["invariance"] is False
    assert {"outcome", "programme", "alignment", "invariance"} <= set(stage_one.terms)
    assert {"outcome", "alignment", "invariance", "programme"} <= set(zero_terms())


def test_the_weights_are_validated() -> None:
    with pytest.raises(ConfigError):
        ObjectiveWeights(lambda_alignment=-1.0).validate()
    with pytest.raises(ConfigError):
        effective_floor(0.8, ObjectiveWeights(lambda_invariance=0.0))
    assert effective_floor(0.8, ObjectiveWeights()).d_min == 0.8
    merged = blend([{"outcome": 1.0}, {"outcome": 2.0, "alignment": 0.5}])
    assert merged["outcome"] == 3.0
    assert merged["alignment"] == 0.5


def test_an_epoch_that_breaches_the_floor_is_reverted() -> None:
    module = torch.nn.Linear(3, 1)
    floor = FeasibilityFloor(d_min=0.8)
    snapshot = state_snapshot(module)
    with torch.no_grad():
        module.weight.add_(1.0)
    verdict = evaluate_epoch(module, 4, 0.5, floor, snapshot)
    assert verdict.admitted is False
    assert verdict.shortfall == pytest.approx(0.3)
    assert torch.allclose(module.weight, snapshot["weight"])
    adopted = evaluate_epoch(module, 5, 0.85, floor, snapshot)
    assert adopted.admitted is True
    assert trade_off_point(True, 0.85, floor)["margin"] == pytest.approx(0.05)


def test_a_ledger_that_never_clears_the_floor_is_reported() -> None:
    floor = FeasibilityFloor(d_min=0.9)
    ledger = FeasibilityLedger(floor=floor)
    ledger.record(0, 0.4)
    ledger.record(1, 0.5)
    assert ledger.n_vetoes() == 2
    worst = ledger.worst()
    assert worst is not None
    assert worst.discrimination == pytest.approx(0.4)
    with pytest.raises(FeasibilityError):
        assert_feasible_at_end(ledger)
    ledger.record(2, 0.95)
    assert_feasible_at_end(ledger)
    assert ledger.summary()["vetoes"] == 2.0
    summary = AttributionSummary(site_accuracy=0.6, region_accuracy=0.5, outcome_auc=0.9)
    flattened = flatten_attribution(summary, 4)
    assert flattened["site_excess_over_chance"] == pytest.approx(0.35)
    assert flattened["site_probe_accuracy"] == pytest.approx(0.6)
