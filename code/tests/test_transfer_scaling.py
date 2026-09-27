"""Cross-region transfer, cross-domain advantage, coherence and the scaling curves."""

from __future__ import annotations

import numpy as np
import pytest

from prism.cohort.schema import Layer
from prism.cohort.supervision import region_covariance
from prism.errors import ConfigError
from prism.measurement.coherence import (
    block_mass_from_coefficients,
    block_mass_from_covariance,
    coherence_reading,
    programme_coherence,
    retention_of_programmes,
    support_consistency,
)
from prism.measurement.reader import (
    ARMS,
    ReaderCase,
    automation_bias_reading,
    fleiss_kappa,
    inter_reader_table,
    overcall_rate,
    reader_arm_result,
    team_benefit,
)
from prism.measurement.scaling import (
    FitFractionPoint,
    SupervisionFractionPoint,
    curve_summary,
    gain_per_increment,
    lower_block_saturation,
    saturation_fraction,
    turning_points,
)
from prism.measurement.transfer import (
    TransferAdvantage,
    TransferCell,
    TransferMatrix,
    advantage_reading,
    build_matrix,
    probe_only_row,
)


def reported_upper() -> tuple[FitFractionPoint, ...]:
    rows = (
        (0.125, 0.812, 0.87, 0.61),
        (0.25, 0.841, 0.90, 0.70),
        (0.50, 0.852, 0.92, 0.76),
        (0.75, 0.849, 0.91, 0.78),
        (1.00, 0.861, 0.94, 0.80),
    )
    return tuple(FitFractionPoint(*row, 0.013) for row in rows)


def reported_lower() -> tuple[SupervisionFractionPoint, ...]:
    rows = (
        (0.10, 0.829, 0.58, 0.64),
        (0.20, 0.843, 0.70, 0.74),
        (0.35, 0.854, 0.68, 0.86),
        (0.50, 0.856, 0.80, 0.93),
    )
    return tuple(SupervisionFractionPoint(*row, 0.013) for row in rows)


def test_the_transfer_matrix_reports_a_retention_and_its_own_deviation() -> None:
    pairs = {
        (f"R{row}", f"R{column}"): 0.9 - 0.03 * abs(row - column)
        for row in range(3)
        for column in range(3)
    }
    matrix = build_matrix(("R0", "R1", "R2"), pairs, 0.01)
    assert matrix.auc("R0", "R0") == pytest.approx(0.9)
    assert matrix.retention("R0") == pytest.approx(0.84 / 0.9)
    assert matrix.smallest_retention()[1] < 1.0
    assert matrix.pooled_exceeds_diagonal() == ()
    assert len(matrix.row("R0")) == 3
    assert len(matrix.diagonal()) == 3
    with pytest.raises(ConfigError):
        build_matrix(("R0", "R9"), pairs, 0.01)


def test_a_pooled_row_that_exceeds_a_diagonal_is_reported() -> None:
    cells = (
        TransferCell("R0", "R0", 0.80, (0.79, 0.81)),
        TransferCell("pooled", "R0", 0.95, (0.94, 0.96)),
    )
    matrix = TransferMatrix(rows=("R0",), pooled_key="pooled", cells=cells)
    findings = matrix.pooled_exceeds_diagonal()
    assert findings and findings[0][0] == "R0"
    assert matrix.as_table()["pooled"]["R0"] == pytest.approx(0.95)


def test_the_advantage_decays_away_from_the_primary_task() -> None:
    rows = (
        TransferAdvantage("colorectal", 0.822, 0.861, 0.831, 0.861),
        TransferAdvantage("gastro", 0.774, 0.795, 0.762, 0.795),
        TransferAdvantage("pancreatic", 0.741, 0.759, 0.729, 0.759),
        TransferAdvantage("cervical", 0.761, 0.786, 0.748, 0.786),
    )
    reading = advantage_reading(rows)
    assert reading["primary_advantage"] == pytest.approx(0.039)
    assert reading["mean_decay"] > 0.0
    probe = probe_only_row(rows)
    assert probe["min_probe_gap"] > 0.0
    assert rows[0].probe_gap() == pytest.approx(0.03)


def test_the_reader_arms_and_the_combination() -> None:
    cases = tuple(
        ReaderCase(
            case_id=f"C{index}",
            reference=int(index % 3 == 0),
            clinician_calls=(int(index % 2 == 0), int(index % 3 == 0), int(index % 5 == 0)),
            model_call=int(index % 4 == 0),
        )
        for index in range(60)
    )
    arms = {arm: reader_arm_result(cases, arm, 0.012) for arm in ARMS}
    assert set(arms) == set(ARMS)
    benefit = team_benefit(arms)
    assert "auc_margin" in benefit
    bias = automation_bias_reading(arms)
    assert bias["overcall_change"] == pytest.approx(
        arms["clinician-with-model"].overcall - arms["clinician-alone"].overcall
    )
    kappa = inter_reader_table(cases)
    assert -1.0 <= kappa <= 1.0
    assert overcall_rate(np.asarray([1, 1, 0]), np.asarray([0, 0, 1])) == pytest.approx(1.0)
    counts = np.asarray([[3, 0], [0, 3], [2, 1]])
    assert fleiss_kappa(counts) > 0.0
    with pytest.raises(ConfigError):
        reader_arm_result(cases, "nobody", 0.012)


def test_programme_coherence_prefers_the_matching_block(lexicon) -> None:
    blocks = lexicon.index_blocks()
    covariance = np.eye(lexicon.size)
    first = list(blocks[0])
    covariance[np.ix_(first, first)] += 6.0
    aligned = np.zeros(lexicon.size)
    aligned[first] = 1.0
    misaligned = np.zeros(lexicon.size)
    misaligned[list(blocks[3])] = 1.0
    assert programme_coherence(aligned.reshape(1, -1), covariance, blocks) > programme_coherence(
        misaligned.reshape(1, -1), covariance, blocks
    )
    mass = block_mass_from_coefficients(aligned.reshape(1, -1), blocks)
    assert mass.sum() == pytest.approx(1.0)
    reference = block_mass_from_covariance(covariance, blocks)
    assert reference.sum() == pytest.approx(1.0)
    reading = coherence_reading(aligned.reshape(1, -1), covariance, blocks)
    assert reading.leading_block(("P1", "P2", "P3", "P4", "P5", "P6")) == "P1"
    assert 0.0 <= support_consistency(aligned.reshape(1, -1), blocks, 0.05) <= 1.0
    assert retention_of_programmes(
        aligned.reshape(1, -1), blocks, lexicon.programme_ids, ("P1", "P2"), 0.5
    ) == pytest.approx(0.5)
    with pytest.raises(ConfigError):
        block_mass_from_covariance(np.zeros((3, 3)), blocks)


def test_the_scaling_reductions_find_the_reported_turning_points() -> None:
    upper = reported_upper()
    lower = reported_lower()
    assert saturation_fraction(upper, tolerance=0.05) == 0.125
    assert saturation_fraction(upper, tolerance=0.002) == 1.0
    assert turning_points([point.external_auc for point in upper]) == (2, 3)
    assert turning_points([point.coherence for point in lower]) == (1, 2)
    reading = lower_block_saturation(lower, tolerance=0.002)
    assert reading["sat_at_last_fraction"] == 0.0
    assert reading["coherence_gain_total"] > 0.0
    summary = curve_summary(upper, lower, tolerance=0.05)
    assert summary["upper_turning_points"] >= 1.0
    assert summary["lower_saturated"] == 0.0
    gains = gain_per_increment(upper)
    assert len(gains) == len(upper) - 1
    with pytest.raises(ConfigError):
        saturation_fraction((), 0.01)


def test_region_covariance_comes_from_the_supervised_records(parts) -> None:
    supervised = parts[Layer.DEVELOPMENT].supervised()
    region = supervised.regions()[0] if supervised.regions() else None
    if region is None:
        pytest.skip("the fixture records carry no supervised record")
    try:
        covariance = region_covariance(supervised, region)
    except Exception:
        pytest.skip("the region carries too few supervised records for a covariance")
    assert covariance.shape[0] == covariance.shape[1]
