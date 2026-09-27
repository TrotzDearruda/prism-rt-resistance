"""Values as printed in the manuscript.

Every number here is transcribed from the article and is kept next to the cell it came from so
that a measured value can be read against it without a second lookup. Nothing in this module is
computed; it is a transcript, and a run that reproduces a row reports both numbers rather than
one of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TABLE1_COLUMNS: tuple[str, ...] = (
    "internal_auc",
    "external_auc",
    "low_resource_auc",
    "calibration_slope",
    "brier",
    "net_benefit",
)

TABLE1_FOOTER: Mapping[str, float] = {
    "internal_auc": 0.008,
    "external_auc": 0.011,
    "low_resource_auc": 0.012,
    "calibration_slope": 0.02,
    "brier": 0.004,
    "net_benefit": 0.005,
}

TABLE1_ROWS: Mapping[str, tuple[float, ...]] = {
    "clinical_nomogram": (0.681, 0.642, 0.618, 0.71, 0.152, 0.061),
    "mri_radiomics": (0.742, 0.689, 0.663, 0.78, 0.141, 0.078),
    "task_specific_mri_deep_learning": (0.798, 0.712, 0.671, 0.82, 0.133, 0.089),
    "radiology_pathology_late_fusion": (0.826, 0.744, 0.698, 0.86, 0.127, 0.096),
    "foundation_model_linear_probe": (0.831, 0.751, 0.703, 0.88, 0.125, 0.098),
    "foundation_model_full_fine_tune": (0.869, 0.769, 0.714, 0.84, 0.117, 0.107),
    "prism_unconstrained": (0.904, 0.822, 0.781, 0.91, 0.104, 0.124),
    "prism_constrained": (0.898, 0.861, 0.842, 0.94, 0.105, 0.129),
    "prism_conditional_on_oxygenation": (0.897, 0.858, 0.840, 0.93, 0.106, 0.127),
}

TABLE1_REPORTED_PREVALENCE = (0.194, 0.182, 0.206)
TABLE1_PRIMARY_PAIR = ("prism_constrained", "prism_unconstrained")
TABLE1_EXTERNAL_MARGIN_OVER_STRONGEST = 0.092

TABLE2_COLUMNS: tuple[str, ...] = (
    "internal_auc",
    "external_auc",
    "lowest_resource_auc",
)

TABLE2_FOOTER: Mapping[str, float] = {
    "internal_auc": 0.008,
    "external_auc": 0.011,
    "lowest_resource_auc": 0.012,
}

TABLE2_ROWS: Mapping[str, tuple[tuple[float, ...], str]] = {
    "full_prism_constrained": ((0.898, 0.861, 0.842), "yes"),
    "minus_alignment": ((0.894, 0.823, 0.788), "partial"),
    "minus_invariance": ((0.907, 0.792, 0.741), "partial"),
    "minus_sparse_signed_decoder": ((0.901, 0.858, 0.838), "no"),
    "minus_interval_head": ((0.898, 0.860, 0.841), "yes"),
    "minus_alignment_and_invariance": ((0.897, 0.734, 0.688), "no"),
    "tier0_permuted_metabolite_alignment": ((0.899, 0.803, 0.762), "no"),
    "tier0_randomised_site_labels": ((0.906, 0.786, 0.735), "partial"),
    "backbone_only_no_constraint": ((0.869, 0.769, 0.714), "no"),
}

TABLE3_REGIONS: tuple[str, ...] = ("Region I", "Region II", "Region III", "Region IV")
TABLE3_POOLED_ROW = "all_regions_site_invariant"

TABLE3_PANEL_A: Mapping[tuple[str, str], float] = {
    ("Region I", "Region I"): 0.909,
    ("Region I", "Region II"): 0.871,
    ("Region I", "Region III"): 0.858,
    ("Region I", "Region IV"): 0.806,
    ("Region II", "Region I"): 0.864,
    ("Region II", "Region II"): 0.897,
    ("Region II", "Region III"): 0.849,
    ("Region II", "Region IV"): 0.817,
    ("Region III", "Region I"): 0.856,
    ("Region III", "Region II"): 0.843,
    ("Region III", "Region III"): 0.889,
    ("Region III", "Region IV"): 0.821,
    ("Region IV", "Region I"): 0.828,
    ("Region IV", "Region II"): 0.834,
    ("Region IV", "Region III"): 0.838,
    ("Region IV", "Region IV"): 0.874,
    (TABLE3_POOLED_ROW, "Region I"): 0.905,
    (TABLE3_POOLED_ROW, "Region II"): 0.874,
    (TABLE3_POOLED_ROW, "Region III"): 0.862,
    (TABLE3_POOLED_ROW, "Region IV"): 0.842,
}

TABLE3_PANEL_A_FOOTER = (0.008, 0.014)
TABLE3_REPORTED_RETENTION = 0.92
TABLE3_REPORTED_POOLED_CELL_HALF_WIDTH = 0.014

TABLE3_PANEL_B_ROWS: Mapping[str, tuple[float, float, float]] = {
    "backbone_only": (0.71, 0.64, 0.869),
    "unconstrained_prism": (0.68, 0.61, 0.904),
    "constrained_variance_penalty": (0.31, 0.29, 0.898),
    "constrained_adversarial_penalty": (0.27, 0.25, 0.894),
}

TABLE3_PANEL_B_FOOTER = (0.03, 0.03, 0.011)

TABLE4_COLUMNS: tuple[str, ...] = (
    "auc",
    "sensitivity_at_fixed_specificity",
    "specificity_at_fixed_sensitivity",
    "overcall_rate",
)

TABLE4_ROWS: Mapping[str, tuple[float, ...]] = {
    "clinician_alone": (0.786, 0.741, 0.712, 0.184),
    "model_alone": (0.861, 0.829, 0.803, 0.141),
    "clinician_with_model_output": (0.874, 0.848, 0.821, 0.127),
}

TABLE4_FOOTER: Mapping[str, float] = {
    "auc": 0.012,
    "sensitivity_at_fixed_specificity": 0.015,
    "specificity_at_fixed_sensitivity": 0.015,
    "overcall_rate": 0.015,
}

TABLE5_PANEL_A: Mapping[str, tuple[str, int, str]] = {
    "TCGA-COAD": ("TCGA-COAD", 461, "not evaluable (no response annotation)"),
    "TCGA-READ": ("TCGA-READ", 172, "not evaluable (no response annotation)"),
    "CPTAC-COAD": ("Vasaikar et al., Cell 2019", 110, "not evaluable (no response annotation)"),
}

TABLE5_PANEL_B: Mapping[str, tuple[str, int, str, float, float]] = {
    "GSE132668": ("GSE132668", 37, "pCR", 0.784, 0.612),
    "GSE119409": ("GSE119409", 66, "response category", 0.751, 0.574),
    "GSE46862": ("GSE46862", 69, "multi-class response", 0.738, 0.551),
    "GSE35452": ("GSE35452", 46, "sensitivity", 0.729, 0.538),
    "GSE45404": ("GSE45404", 80, "resistance category", 0.762, 0.583),
    "GSE68204": ("GSE68204", 125, "pCR", 0.771, 0.596),
}

TABLE5_FOOTER: Mapping[str, float] = {"auc": 0.020, "auprc": 0.024}
TABLE5_VERIFICATION_DATE = "2026-09-21"

TABLE6_COLUMNS: tuple[str, ...] = (
    "unconstrained_auc",
    "constrained_auc",
    "advantage",
    "linear_probe_auc",
    "metabolic_supervision_auc",
)

TABLE6_ROWS: Mapping[str, tuple[float, ...]] = {
    "colorectal_primary": (0.822, 0.861, 0.039, 0.831, 0.861),
    "adjacent_gastro_oesophageal": (0.774, 0.795, 0.021, 0.762, 0.795),
    "adjacent_pancreatic": (0.741, 0.759, 0.018, 0.729, 0.759),
    "non_colorectal_cervical": (0.761, 0.786, 0.025, 0.748, 0.786),
}

TABLE6_FOOTER: Mapping[str, float] = {
    "unconstrained_auc": 0.015,
    "constrained_auc": 0.015,
    "advantage": 0.008,
    "linear_probe_auc": 0.015,
    "metabolic_supervision_auc": 0.015,
}

TABLE6_ADJACENT_ADVANTAGES = (0.021, 0.018, 0.014)

TABLE7_UPPER_FRACTIONS: tuple[float, ...] = (0.125, 0.25, 0.50, 0.75, 1.00)
TABLE7_UPPER_ROWS: Mapping[float, tuple[float, float, float]] = {
    0.125: (0.812, 0.87, 0.61),
    0.25: (0.841, 0.90, 0.70),
    0.50: (0.852, 0.92, 0.76),
    0.75: (0.849, 0.91, 0.78),
    1.00: (0.861, 0.94, 0.80),
}
TABLE7_UPPER_FOOTER: Mapping[str, float] = {
    "external_auc": 0.013,
    "calibration_slope": 0.02,
    "programme_coherence": 0.03,
}

TABLE7_LOWER_FRACTIONS: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50)
TABLE7_LOWER_ROWS: Mapping[float, tuple[float, float, float]] = {
    0.10: (0.829, 0.58, 0.64),
    0.20: (0.843, 0.70, 0.74),
    0.35: (0.854, 0.68, 0.86),
    0.50: (0.856, 0.80, 0.93),
}
TABLE7_LOWER_FOOTER: Mapping[str, float] = {
    "external_auc": 0.013,
    "programme_coherence": 0.03,
    "retention": 0.02,
}
TABLE7_RETAINED_PROGRAMMES: tuple[str, ...] = ("P1", "P2", "P3", "P4")

TABLE8_COLUMNS: tuple[str, ...] = (
    "external_auc",
    "calibration_slope",
    "mean_interval_width",
    "empirical_coverage",
)

TABLE8_ROWS: Mapping[str, tuple[float, ...]] = {
    "single_initialisation": (0.857, 0.90, 0.041, 0.884),
    "five_member_seed_ensemble": (0.861, 0.94, 0.033, 0.897),
    "ten_member_seed_ensemble": (0.862, 0.95, 0.031, 0.899),
    "regime_ensemble_probe_constrained": (0.873, 0.93, 0.036, 0.891),
}

TABLE8_FOOTER: Mapping[str, float] = {
    "external_auc": 0.012,
    "calibration_slope": 0.02,
    "mean_interval_width": 0.003,
    "empirical_coverage": 0.008,
}

SUCCESS_MARGIN = 0.015
PRIMARY_DIFFERENCE = -0.006
EXTERNAL_ADVANTAGE = 0.039
NULL_BAND = 0.003
MULTIPLICITY_FAMILY = 5

SYNERGY_COSTS = {"alignment": 0.038, "invariance": 0.069, "both": 0.107}
SYNERGY_JOINT_COST_QUOTED = 0.127

SENSITIVITY_ANALYSIS_EXTERNAL_DROP = 0.055

READER_STUDY_AGREEMENT = 0.714

CONDITIONAL_PROXY_SHIFT = {"external_from": 0.861, "external_to": 0.858}

CROSS_SETTING_DISCRIMINATION_GAP = 0.030


@dataclass(frozen=True)
class ReportedRow:
    """One transcribed row, addressable by table and key."""

    table: str
    key: str
    columns: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "table": self.table,
            "key": self.key,
            "columns": list(self.columns),
            "values": list(self.values),
        }


def table1_rows() -> tuple[ReportedRow, ...]:
    return tuple(
        ReportedRow("table1", key, TABLE1_COLUMNS, values) for key, values in TABLE1_ROWS.items()
    )


def table2_rows() -> tuple[ReportedRow, ...]:
    return tuple(
        ReportedRow("table2", key, TABLE2_COLUMNS, values)
        for key, (values, _) in TABLE2_ROWS.items()
    )


def table3_panel_b_rows() -> tuple[ReportedRow, ...]:
    columns = ("site_probe_accuracy", "region_probe_accuracy", "outcome_auc_retained")
    return tuple(
        ReportedRow("table3b", key, columns, values) for key, values in TABLE3_PANEL_B_ROWS.items()
    )


def table4_rows() -> tuple[ReportedRow, ...]:
    return tuple(
        ReportedRow("table4", key, TABLE4_COLUMNS, values) for key, values in TABLE4_ROWS.items()
    )


def table6_rows() -> tuple[ReportedRow, ...]:
    return tuple(
        ReportedRow("table6", key, TABLE6_COLUMNS, values) for key, values in TABLE6_ROWS.items()
    )


def table7_upper_rows() -> tuple[ReportedRow, ...]:
    columns = ("external_auc", "calibration_slope", "programme_coherence")
    return tuple(
        ReportedRow("table7-upper", f"{fraction:.3f}", columns, values)
        for fraction, values in TABLE7_UPPER_ROWS.items()
    )


def table7_lower_rows() -> tuple[ReportedRow, ...]:
    columns = ("external_auc", "programme_coherence", "retention")
    return tuple(
        ReportedRow("table7-lower", f"{fraction:.2f}", columns, values)
        for fraction, values in TABLE7_LOWER_ROWS.items()
    )


def table8_rows() -> tuple[ReportedRow, ...]:
    return tuple(
        ReportedRow("table8", key, TABLE8_COLUMNS, values) for key, values in TABLE8_ROWS.items()
    )


def all_reported_rows() -> tuple[ReportedRow, ...]:
    collected: list[ReportedRow] = []
    for builder in (
        table1_rows,
        table2_rows,
        table3_panel_b_rows,
        table4_rows,
        table6_rows,
        table7_upper_rows,
        table7_lower_rows,
        table8_rows,
    ):
        collected.extend(builder())
    return tuple(collected)


def cell(table: str, key: str, column: str) -> float:
    """One transcribed cell, by table, row key and column name."""
    lookups = {
        "table1": (TABLE1_COLUMNS, TABLE1_ROWS),
        "table2": (TABLE2_COLUMNS, {k: v[0] for k, v in TABLE2_ROWS.items()}),
        "table4": (TABLE4_COLUMNS, TABLE4_ROWS),
        "table6": (TABLE6_COLUMNS, TABLE6_ROWS),
        "table8": (TABLE8_COLUMNS, TABLE8_ROWS),
    }
    if table not in lookups:
        raise KeyError(f"table {table!r} is not transcribed as a flat grid")
    columns, rows = lookups[table]
    if key not in rows:
        raise KeyError(f"table {table!r} has no row {key!r}")
    if column not in columns:
        raise KeyError(f"table {table!r} has no column {column!r}")
    return float(rows[key][columns.index(column)])


def rows_with_column(table: str, column: str) -> Sequence[tuple[str, float]]:
    """One column of a transcribed table, in the order the rows were declared."""
    lookups = {
        "table1": (TABLE1_COLUMNS, TABLE1_ROWS),
        "table2": (TABLE2_COLUMNS, {k: v[0] for k, v in TABLE2_ROWS.items()}),
        "table4": (TABLE4_COLUMNS, TABLE4_ROWS),
        "table6": (TABLE6_COLUMNS, TABLE6_ROWS),
        "table8": (TABLE8_COLUMNS, TABLE8_ROWS),
    }
    columns, rows = lookups[table]
    index = columns.index(column)
    return tuple((key, float(values[index])) for key, values in rows.items())


def conditional_proxy_gain() -> float:
    """Movement of external discrimination when the oxygenation proxy is entered.

    Ref: Results Sec. 2.1 -- the proxy changes external discrimination from 0.861 to 0.858, and
    the direction of that residual is reported as the genuine outcome of the conditional test.
    """
    return CONDITIONAL_PROXY_SHIFT["external_to"] - CONDITIONAL_PROXY_SHIFT["external_from"]
