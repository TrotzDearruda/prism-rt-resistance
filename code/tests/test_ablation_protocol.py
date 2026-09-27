"""The ablation matrix, the analysis plan and the table documents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prism.assessment.ablation import (
    ABLATIONS,
    arm_for,
    configuration_for,
    control_recovery,
    interpretability_reading,
    permute_metabolite_rows,
    randomise_site_labels,
    table_rows,
)
from prism.assessment.conditional import (
    conditional_gain,
    conditional_table,
    fitted_auc,
    programme_effect_columns,
    proxy_shift,
    reading_summary,
)
from prism.assessment.harness import (
    bootstrap_auc,
    column_variability_table,
    paired_contrast_row,
    table1_predictors,
    table2_ablation,
    table3_transfer,
    table4_reader,
    table5_auxiliary,
    table6_cross_domain,
    table7_scaling,
    table8_ensemble,
)
from prism.assessment.protocol import (
    adjudicate_plan,
    criterion_from_measurements,
    default_plan,
    plan_digest,
    report_references,
)
from prism.assessment.reporting import (
    TableDocument,
    compare_to_reference,
    footer_note,
    note_measurement_origin,
    summarise_documents,
    with_footer_note,
    write_bundle,
    write_document,
)
from prism.cohort.pathways import INCONCLUSIVE, LOSES, RETAINS, conditional_verdicts
from prism.errors import ConfigError, ProtocolError
from prism.measurement.attribution import attribution_row
from prism.measurement.reader import ARMS, ReaderCase, reader_arm_result
from prism.statistics.family import FAMILY
from tests.conftest import tiny_overrides


def test_every_ablation_row_maps_onto_a_configuration() -> None:
    base = tiny_overrides()
    assert len(ABLATIONS) == 9
    assert sum(1 for arm in ABLATIONS if arm.is_tier_zero()) == 2
    for arm in ABLATIONS:
        config = configuration_for(base, arm.key)
        assert config.arm == arm.arm
        if arm.zeroes_alignment:
            assert config.objective.lambda_alignment == 0.0
        if arm.zeroes_invariance:
            assert config.objective.lambda_invariance == 0.0
    with pytest.raises(ConfigError):
        arm_for("no such row")


def test_the_tier_zero_controls_break_the_pairing_without_breaking_the_histogram() -> None:
    rng = np.random.default_rng(5)
    metabolites = rng.normal(size=(48, 10))
    regions = tuple(["Region I"] * 24 + ["Region II"] * 24)
    permuted = permute_metabolite_rows(metabolites, seed=3, regions=regions)
    assert not np.allclose(metabolites, permuted)
    assert sorted(metabolites[:24, 0].tolist()) == sorted(permuted[:24, 0].tolist())
    sites = np.asarray(["A"] * 24 + ["B"] * 24, dtype=object)
    shuffled = randomise_site_labels(sites, seed=9)
    assert sorted(sites.tolist()) == sorted(shuffled.tolist())
    assert not np.array_equal(sites, shuffled)


def test_the_controls_recover_the_constraint_effect() -> None:
    reading = control_recovery(full_value=0.861, control_value=0.803, unconstrained_value=0.822)
    assert reading["constraint_effect"] == pytest.approx(0.039)
    assert reading["recovered_share"] > 0.0
    rows = table_rows({arm.key: (0.898, 0.861, 0.842) for arm in ABLATIONS})
    assert len(rows) == 9
    reading_of_rows = interpretability_reading(rows)
    assert reading_of_rows["n_rows"] == 9.0
    assert 0.0 < reading_of_rows["interpretable_fraction"] < 1.0
    assert reading_of_rows["n_partial"] > 0.0
    with pytest.raises(ConfigError):
        table_rows({"full_prism_constrained": (0.9, 0.8, 0.7)})


def test_the_conditional_rule_maps_the_three_outcomes() -> None:
    verdicts = conditional_verdicts(
        {"P1": 0.05, "P2": 0.004, "P3": 0.0, "P4": -0.02, "P5": 0.0031, "P6": 0.0029}, 0.003
    )
    assert verdicts["P1"] == RETAINS
    assert verdicts["P2"] == RETAINS
    assert verdicts["P3"] == LOSES
    assert verdicts["P4"] == LOSES
    assert verdicts["P6"] == INCONCLUSIVE
    table = conditional_table(
        {"P1": 0.05, "P2": -0.01, "P3": 0.0, "P4": 0.0, "P5": 0.0, "P6": 0.0}, 0.003
    )
    summary = reading_summary(table, 0.003)
    assert summary["n_retained"] == 1.0
    assert summary["n_lost"] >= 1.0


def test_the_conditional_gain_is_read_out_of_sample() -> None:
    rng = np.random.default_rng(41)
    n_rows = 900
    labels = (rng.random(n_rows) < 0.35).astype(int)
    proxy = labels * 1.2 + rng.normal(0.0, 1.0, size=n_rows)
    informative = labels * 1.1 + rng.normal(0.0, 1.0, size=n_rows)
    inert = rng.normal(0.0, 1.0, size=n_rows)
    strong = conditional_gain(proxy, proxy, informative, labels)
    weak = conditional_gain(proxy, proxy, inert, labels)
    assert strong > weak
    assert strong > 0.01
    assert abs(weak) < 0.05
    columns = programme_effect_columns(({"P1": 0.2}, {"P2": -0.1}), ("P1", "P2"))
    assert columns["P1"].tolist() == [0.2, 0.0]
    assert proxy_shift(0.861, 0.858)["external_delta"] == pytest.approx(-0.003)
    design = np.column_stack([np.ones(200), np.linspace(-2, 2, 200)])
    truth = (design[:, 1] > 0).astype(float)
    assert fitted_auc(design, truth) > 0.9


def test_the_analysis_plan_is_validated_and_digested() -> None:
    plan = default_plan(0.015, 0.003, 0.8, 0.90)
    assert len(plan.family) == len(FAMILY) == 5
    digest = plan_digest(plan)
    assert len(digest) == 64
    assert plan_digest(plan) == digest
    references = report_references(plan)
    assert set(references) == {"clinical-standard-of-care", "convolutional-benchmark"}
    with pytest.raises(ProtocolError):
        default_plan(-1.0, 0.003, 0.8, 0.90)
    with pytest.raises(ProtocolError):
        default_plan(0.015, 0.003, 0.8, 0.90, references=("only-one",))
    criterion = criterion_from_measurements(0.898, 0.904, 0.861, 0.822, 0.015)
    assert criterion.satisfied()
    outcome = adjudicate_plan(
        plan,
        {spec.name: 0.01 for spec in FAMILY},
        primary_constrained=0.898,
        primary_unconstrained=0.904,
        external_constrained=0.861,
        external_unconstrained=0.822,
        alignment_cost=0.038,
        invariance_cost=0.069,
        both_cost=0.127,
        conditional_gain={"P1": 0.05, "P2": 0.02, "P3": 0.01, "P4": -0.01, "P5": 0.0, "P6": 0.0},
        retained=("P1", "P2", "P3"),
    )
    body = outcome.as_dict()
    assert body["criterion"]["satisfied"] is True
    assert body["synergy"]["super_additive_excess"] == pytest.approx(0.02)
    assert any(entry["rejected"] for entry in body["family"])


def test_a_table_document_needs_its_variability_row() -> None:
    document = TableDocument(
        name="t",
        caption="a caption",
        columns=("predictor", "value"),
        rows=({"predictor": "a", "value": 1.0},),
        footer={"value": 0.01},
    )
    document.validate()
    assert "0.01" in footer_note(document.footer, document.coverage_level)
    annotated = with_footer_note(document)
    assert len(annotated.notes) == 1
    assert note_measurement_origin(annotated, "measured on the staged arm").notes[-1] == (
        "measured on the staged arm"
    )
    with pytest.raises(ConfigError):
        TableDocument(
            name="t2", caption="c", columns=("a",), rows=({"a": 1.0},), footer={}
        ).validate()
    summary = summarise_documents((document,))
    assert summary["n_documents"] == 1
    comparison = compare_to_reference({"a": 0.9}, {"a": 0.95}, 0.1)
    assert comparison["a"]["within_tolerance"] is True


def test_documents_round_trip_through_disk(tmp_path: Path) -> None:
    document = TableDocument(
        name="tablex",
        caption="a caption",
        columns=("row", "value"),
        rows=({"row": "a", "value": 0.5}, {"row": "b", "value": 0.6}),
        footer={"value": 0.02},
    )
    target = write_document(tmp_path, document)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["name"] == "tablex"
    assert payload["footer"]["value"] == pytest.approx(0.02)
    assert payload["notes"]
    digests = write_bundle(tmp_path, (document,))
    assert digests["tablex"]


def test_the_table_assemblers_require_their_columns() -> None:
    rng = np.random.default_rng(3)
    n_rows = 200
    labels = (rng.random(n_rows) < 0.3).astype(int)
    scores = labels * 1.2 + rng.normal(0.0, 1.0, size=n_rows)
    blocks = {"internal": scores, "external": scores, "low_resource": scores}
    predictors = {
        "prism_constrained": blocks,
        "foundation_model_linear_probe": {"internal": scores},
    }
    table = table1_predictors(
        {"prism_constrained": blocks},
        {"internal": labels, "external": labels, "low_resource": labels},
        (0.1, 0.2, 0.3),
    )
    assert table.name == "table1"
    assert table.footer
    with pytest.raises(ConfigError):
        table1_predictors(predictors, {"internal": labels}, (0.1,))
    ablation = table2_ablation(
        {"full": (0.9, 0.86, 0.84), "minus_alignment": (0.89, 0.82, 0.78)},
        {"full": "yes", "minus_alignment": "partial"},
        {"internal_auc": 0.008, "external_auc": 0.011, "lowest_resource_auc": 0.012},
    )
    assert len(ablation.rows) == 2
    features = rng.normal(size=(120, 6))
    sites = np.asarray([f"S{index % 3}" for index in range(120)], dtype=object)
    regions = np.asarray([f"R{index % 3}" for index in range(120)], dtype=object)
    panel_a, panel_b = table3_transfer(
        rows=("Region I",),
        auc_by_pair={("Region I", "Region I"): 0.9},
        cell_half_widths={("Region I", "Region I"): 0.01},
        attribution=(attribution_row("constrained", features, sites, regions, 0.9),),
    )
    assert panel_a.name == "table3-panel-a"
    assert panel_b.name == "table3-panel-b"


def test_the_remaining_assemblers_build_documents() -> None:
    cases = tuple(
        ReaderCase(f"C{i}", int(i % 3 == 0), (int(i % 2 == 0), 0, 1), int(i % 4 == 0))
        for i in range(60)
    )
    arms = {arm: reader_arm_result(cases, arm, 0.012) for arm in ARMS}
    reader_table = table4_reader(arms, 0.5, {"auc": 0.012})
    assert reader_table.name == "table4"
    auxiliary = table5_auxiliary(
        {"TCGA-COAD": {"accession": "TCGA-COAD", "n": 461, "endpoint": "not evaluable"}},
        {
            "GSE132668": {
                "accession": "GSE132668",
                "n": 37,
                "endpoint": "pCR",
                "auc": 0.784,
                "auprc": 0.612,
            }
        },
        {"auc": 0.02, "auprc": 0.024},
    )
    assert len(auxiliary.rows) == 2
    cross = table6_cross_domain(
        {
            "colorectal": {
                "unconstrained_auc": 0.822,
                "constrained_auc": 0.861,
                "advantage": 0.039,
                "linear_probe_auc": 0.831,
                "metabolic_supervision_auc": 0.861,
            }
        },
        {"external_auc": 0.015},
    )
    assert cross.name == "table6"
    upper, lower = table7_scaling(
        (
            {
                "fraction_of_arm": 0.125,
                "external_auc": 0.812,
                "calibration_slope": 0.87,
                "programme_coherence": 0.61,
            },
        ),
        (
            {
                "fraction_supervised": 0.10,
                "external_auc": 0.829,
                "programme_coherence": 0.58,
                "retention_of_named_subset": 0.64,
            },
        ),
        {"external_auc": 0.013},
        {"external_auc": 0.013},
    )
    assert upper.name == "table7-upper"
    assert lower.name == "table7-lower"
    ensemble = table8_ensemble(
        (
            {
                "configuration": "single",
                "external_auc": 0.857,
                "calibration_slope": 0.90,
                "mean_interval_width": 0.041,
                "empirical_coverage": 0.884,
            },
        ),
        {"external_auc": 0.012},
    )
    assert ensemble.name == "table8"
    assert column_variability_table({"value": [0.1, 0.2, 0.3]})[0].column == "value"


def test_a_paired_contrast_row_reports_the_family_columns() -> None:
    rng = np.random.default_rng(9)
    n_rows = 200
    labels = (rng.random(n_rows) < 0.3).astype(int)
    strong = labels * 1.3 + rng.normal(0.0, 1.0, size=n_rows)
    weak = labels * 0.4 + rng.normal(0.0, 1.0, size=n_rows)
    row = paired_contrast_row(strong, weak, labels, "a-vs-b")
    assert row["contrast"] == "a-vs-b"
    assert float(row["difference"]) > 0.0
    assert bootstrap_auc is not None
