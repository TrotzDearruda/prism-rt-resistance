"""The reported programme, its effects and the intervals around them."""

from __future__ import annotations

import numpy as np
import pytest

from prism.cohort.schema import Layer
from prism.emission.intervals import (
    cohort_site_counts,
    coverage_of,
    fold_coverages,
    interval_arrays_from_columns,
    interval_from_effect,
    leave_one_site_out_folds,
    pooled_coverage,
    shift_reading,
)
from prism.emission.programmes import (
    ProgrammeStatement,
    effect_table,
    emission_diagnostics,
    emit_programmes,
    programme_ranking,
    restricted_to,
    sign_agreement_of_statements,
    statement_table,
    support_summary,
)
from prism.errors import ConfigError
from prism.modelling.programme_head import SupportThreshold


def statements(lexicon, n_rows: int = 24, seed: int = 7) -> tuple[ProgrammeStatement, ...]:
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(0.0, 0.4, size=(n_rows, lexicon.size))
    beta = rng.normal(0.0, 0.5, size=(n_rows, lexicon.n_blocks()))
    scale = np.abs(rng.normal(0.2, 0.05, size=beta.shape))
    return emit_programmes(
        coefficients,
        tuple(f"R{index:06d}" for index in range(n_rows)),
        lexicon,
        beta,
        scale,
        nominal_coverage=0.90,
        threshold=SupportThreshold(0.15, 1, 4),
    )


def test_emission_is_a_signed_programme_with_bounded_support(lexicon) -> None:
    emitted = statements(lexicon)
    assert len(emitted) == 24
    for statement in emitted:
        assert 6 <= len(statement.supported_metabolites) <= 24
        assert set(statement.signs) == set(statement.supported_metabolites)
        assert set(statement.effects) == set(statement.supported_programmes)
        assert set(statement.intervals) == set(statement.supported_programmes)
        for low, high in statement.intervals.values():
            assert low < high


def test_effects_and_interval_widths_are_reported_per_programme(lexicon) -> None:
    emitted = statements(lexicon)
    table = effect_table(emitted, lexicon.programme_ids)
    assert table
    for entry in table.values():
        assert entry["n_observations"] > 0
        assert entry["mean_interval_width"] > 0.0
        assert entry["sd_effect"] >= 0.0
    width = emitted[0].interval_width(emitted[0].supported_programmes[0])
    assert width > 0.0
    ratio = emitted[0].effect_over_width(emitted[0].supported_programmes[0])
    assert ratio >= 0.0


def test_the_ranking_and_the_support_summary_are_consistent(lexicon) -> None:
    emitted = statements(lexicon)
    ranked = programme_ranking(emitted)
    assert ranked
    assert all(isinstance(name, str) for name, _ in ranked)
    summary = support_summary(emitted)
    assert 6.0 <= summary["mean_supported_metabolites"] <= 24.0
    assert 1.0 <= summary["mean_supported_programmes"] <= 6.0
    table = statement_table(emitted[:2])
    assert len(table) == 2
    assert "supported_metabolites" in table[0]
    with pytest.raises(ConfigError):
        support_summary(())


def test_signs_agree_with_the_reported_directions(lexicon) -> None:
    emitted = statements(lexicon)
    agreement = sign_agreement_of_statements(emitted, lexicon)
    assert 0.0 <= agreement <= 1.0


def test_a_statement_can_be_restricted_to_named_programmes(lexicon) -> None:
    emitted = statements(lexicon)
    restricted = restricted_to(emitted[0], lexicon, ("P1",))
    assert set(restricted.supported_programmes) <= {"P1"}
    members = {lexicon.names()[index] for index in lexicon.block_members("P1")}
    assert set(restricted.supported_metabolites) <= members
    assert set(restricted.signs) <= members


def test_emission_diagnostics_read_the_support(lexicon) -> None:
    rng = np.random.default_rng(11)
    coefficients = rng.normal(0.0, 0.4, size=(6, lexicon.size))
    expected = {entry.name: entry.expected_sign for entry in lexicon.catalogue()}
    diagnostics = emission_diagnostics(coefficients, lexicon, expected)
    assert "support_overall" in diagnostics
    assert 0.0 <= diagnostics["sign_agreement"] <= 1.0


def test_intervals_are_built_from_effects_and_scales() -> None:
    effects = np.asarray([[0.2, -0.1]])
    scales = np.asarray([[0.05, 0.1]])
    lower, upper = interval_from_effect(effects, scales, 0.90)
    assert lower.shape == upper.shape == effects.shape
    assert float(lower[0, 0]) < 0.2 < float(upper[0, 0])
    with pytest.raises(ConfigError):
        interval_from_effect(effects, np.asarray([0.1]), 0.90)
    coverage, width = coverage_of(lower, upper, np.asarray([[0.2, -0.1]]))
    assert coverage == 1.0
    assert width > 0.0


def test_leave_one_site_out_folds_and_pooled_coverage() -> None:
    sites = ("A", "B", "C")
    per_site = {"A": 6, "B": 4, "C": 5}
    folds = leave_one_site_out_folds(sites, per_site)
    assert len(folds) == 3
    for _site, fitting, held in folds:
        assert fitting.size + held.size == sum(per_site.values())
        assert set(fitting).isdisjoint(set(held.tolist()))
    lower = np.zeros(15)
    upper = np.ones(15)
    targets = np.full(15, 0.5)
    labels = ["A"] * 6 + ["B"] * 4 + ["C"] * 5
    rows = fold_coverages(lower, upper, targets, labels)
    assert len(rows) == 3
    pooled = pooled_coverage(rows, 0.90)
    assert pooled["pooled_coverage"] == pytest.approx(1.0)
    assert shift_reading(pooled, pooled, 0.003)["coverage_failure"] == 0.0
    assert cohort_site_counts_from_folds(rows) == 15


def cohort_site_counts_from_folds(rows) -> int:
    return sum(row.n_intervals for row in rows)


def test_interval_columns_line_up_with_the_programme_order() -> None:
    effects = {"P1": 0.2, "P2": -0.1}
    spreads = {"P1": 0.05, "P2": 0.1}
    intervals, widths = interval_arrays_from_columns(effects, spreads, 0.90)
    assert set(intervals) == {"P1", "P2"}
    assert widths["P2"] > widths["P1"]
    with pytest.raises(ConfigError):
        interval_arrays_from_columns(effects, {"P1": 0.05}, 0.90)


def test_site_counts_are_read_from_a_cohort(parts) -> None:
    counts = cohort_site_counts(parts[Layer.DEVELOPMENT])
    assert sum(counts.values()) == len(parts[Layer.DEVELOPMENT])
