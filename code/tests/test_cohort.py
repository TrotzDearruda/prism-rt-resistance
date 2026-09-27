"""Cohort assembly, partitioning, prevalence and supervision."""

from __future__ import annotations

import pytest

from prism.cohort.archive import describe_archive, read_archive, write_archive
from prism.cohort.lexicon import default_lexicon
from prism.cohort.partitions import (
    PartitionSpec,
    check_no_patient_overlap,
    check_site_boundary,
    fold_by_region,
    fold_by_site,
    stratified_supervision_frame,
    training_frame,
)
from prism.cohort.prevalence import (
    prevalence_matches_report,
    reported_prevalence_interval,
    wilson,
)
from prism.cohort.schema import Layer
from prism.cohort.strata import STRATUM_MODALITIES, default_registry, weakest_stratum
from prism.cohort.supervision import (
    alignment_batches,
    coverage_report,
    region_covariance,
    supervised_index,
)
from prism.errors import CohortError, SupervisionError
from prism.numerics.operating_point import RuleInProfile


def test_lexicon_partitions_the_vocabulary(lexicon) -> None:
    flat = [index for block in lexicon.index_blocks() for index in block]
    assert sorted(flat) == list(range(lexicon.size))
    assert lexicon.n_blocks() == 6
    assert len(lexicon.expected_signs()) == lexicon.size
    assert set(lexicon.expected_signs()) == {-1, 1}


def test_default_registry_matches_the_reported_settings() -> None:
    registry = default_registry()
    assert registry.names() == ("A", "B", "C", "D", "E", "F", "G")
    assert len(registry.regions()) == 4
    assert len(registry.strata()) == 3
    assert weakest_stratum(registry) == "lower-middle-resource"
    assert len(STRATUM_MODALITIES["high-resource"]) == 4


def test_partitioning_keeps_patients_whole(parts) -> None:
    check_no_patient_overlap(parts)
    check_site_boundary(
        parts,
        PartitionSpec(
            development_patients=36,
            internal_patients=20,
            external_patients=18,
            silent_patients=0,
            development_sites=("A", "B", "C", "D", "E"),
            holdout_sites=("F", "G"),
            external_regions=("Region III", "Region IV"),
        ),
    )
    assert len(parts[Layer.INTERNAL]) > 0
    assert len(parts[Layer.EXTERNAL]) > 0


def test_a_patient_keeps_one_setting_and_one_response(records) -> None:
    seen: dict[str, tuple[str, int]] = {}
    for record in records.cohort:
        previous = seen.get(record.patient_id)
        if previous is None:
            seen[record.patient_id] = (record.site, record.outcome)
            continue
        assert previous == (record.site, record.outcome)


def test_a_staged_arm_reads_back_as_it_was_written(records, tmp_path) -> None:
    archive = tmp_path / "cohort.npz"
    write_archive(
        records.cohort,
        archive,
        metabolite_width=records.lexicon.size,
        provenance="staged by the test suite",
    )
    contents = describe_archive(archive)
    assert contents.n_records == len(records.cohort)
    assert contents.metabolite_width == records.lexicon.size
    assert contents.provenance == "staged by the test suite"
    assert set(contents.layer_counts) <= {layer.value for layer in Layer}
    assert contents.layer_counts.get("development", 0) > 0
    staged = read_archive(archive)
    assert len(staged) == len(records.cohort)
    assert staged.patient_ids() == records.cohort.patient_ids()
    assert staged.site_labels().tolist() == records.cohort.site_labels().tolist()
    assert len(staged.supervised()) == len(records.cohort.supervised())
    for original, restored in zip(records.cohort, staged, strict=True):
        assert set(restored.availability) == set(original.availability)


def test_an_absent_or_mismatched_arm_is_reported(records, tmp_path) -> None:
    with pytest.raises(CohortError):
        read_archive(tmp_path / "absent.npz")
    archive = tmp_path / "cohort.npz"
    write_archive(records.cohort, archive, metabolite_width=records.lexicon.size)
    with pytest.raises(CohortError):
        read_archive(archive, expected_shapes={"ct": (9, 9, 9, 9)})
    with pytest.raises(CohortError):
        read_archive(archive, expected_shapes={"fluoroscopy": (4,)})


def test_prevalence_reconstruction_reproduces_the_printed_bounds() -> None:
    interval = reported_prevalence_interval()
    assert interval.estimate == pytest.approx(0.194, abs=1e-3)
    assert interval.lower == pytest.approx(0.182, abs=2e-3)
    assert interval.upper == pytest.approx(0.206, abs=2e-3)
    assert prevalence_matches_report(tolerance=2e-3)
    wo = wilson(0.194, 3847)
    assert wo.lower < interval.estimate < wo.upper


def test_rule_in_value_falls_below_one_at_the_deployment_rate() -> None:
    profile = RuleInProfile(prevalence=0.194, sensitivity=0.829, specificity=0.803)
    assert profile.ppv() < 1.0
    assert 0.0 < profile.npv() < 1.0
    assert profile.odds_ratio() > 1.0


def test_supervision_coverage_and_alignment_batches(records) -> None:
    report = coverage_report(records.cohort)
    assert report.supervised_records > 0
    assert 0.0 < report.fraction() <= 1.0
    positions, regions = supervised_index(records.cohort)
    assert positions.size == report.supervised_records
    assert len(regions) == positions.size
    draws = list(alignment_batches(records.cohort, 12, seed=4, max_batches=2))
    assert draws
    for draw in draws:
        assert draw.metabolite.shape[0] == draw.rows.size
        assert draw.metabolite.shape[1] == default_lexicon().size
    for region in records.cohort.regions():
        try:
            covariance = region_covariance(records.cohort, region)
        except SupervisionError:
            continue
        assert covariance.shape == (default_lexicon().size, default_lexicon().size)


def test_folds_hold_a_region_or_a_site_out(parts) -> None:
    cohort = parts[Layer.DEVELOPMENT]
    for site, rest, held in fold_by_site(cohort):
        assert site not in rest.sites()
        assert set(held.sites()) == {site}
    for region, rest, held in fold_by_region(cohort):
        assert region not in rest.regions()
        assert set(held.regions()) == {region}


def test_scaling_frames_shrink_without_breaking_patients(parts) -> None:
    cohort = parts[Layer.DEVELOPMENT]
    less = training_frame(cohort, 0.5, seed=2)
    assert len(less) < len(cohort)
    patients = set(less.patient_ids())
    assert patients <= set(cohort.patient_ids())
    supervised = cohort.supervised()
    assert len(supervised) > 0
    thinned = stratified_supervision_frame(cohort, 0.5, seed=2)
    assert len(thinned.supervised()) <= len(supervised)


def test_an_impossible_request_is_refused(records) -> None:
    with pytest.raises(CohortError):
        training_frame(records.development(), 0.0, seed=1)
    with pytest.raises(CohortError):
        training_frame(records.development(), 1.5, seed=1)


def test_availability_follows_the_resource_stratum(parts) -> None:
    cohort = parts[Layer.DEVELOPMENT]
    for record in cohort:
        expected = STRATUM_MODALITIES[record.stratum]
        assert record.availability == expected
        for modality in expected:
            assert record.has(modality)
    for unavailable in ("mri", "histology"):
        weakest = cohort.by_stratum("lower-middle-resource")
        if len(weakest) == 0:
            continue
        assert all(not record.has(unavailable) for record in weakest)
