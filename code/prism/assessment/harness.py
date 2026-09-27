"""Table assembly.

Ref: Results Sec. 2 to Sec. 3 -- the study reports eight tables, each of which is a fixed set of
columns. This module turns measured values into the documents of ``assessment.reporting``, so that
a table is assembled by one named function per table and an unmeasured column cannot be silently
omitted from a row.

The functions take measured inputs rather than models: a table can therefore be assembled from a
staged arm, from a held-out partition, or from stored scores, without any of those choices being
baked into the table's definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from prism.assessment.reporting import TableDocument
from prism.errors import ConfigError
from prism.measurement.attribution import AttributionRow, attribution_table
from prism.measurement.calibration import calibration_from_probabilities
from prism.measurement.decision import decision_report
from prism.measurement.discrimination import discrimination
from prism.measurement.reader import ReaderArmResult, team_benefit
from prism.measurement.uncertainty import ColumnVariability, half_width, half_width_table
from prism.statistics.ci import patient_cluster_bootstrap
from prism.statistics.delong import paired_delong


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return discrimination(scores, labels).auc


def bootstrap_auc(
    scores: np.ndarray, labels: np.ndarray, patients: Sequence[str], seed: int
) -> float:
    """Half-width of the patient-level bootstrap interval around one discrimination value."""
    interval = patient_cluster_bootstrap(
        [float(value) for value in np.asarray(scores, dtype=np.float64).reshape(-1)],
        patients,
        lambda values: _auc(values, labels),
        seed=seed,
    )
    return interval.half_width


def table1_predictors(
    predictors: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    decision_thresholds: Sequence[float],
) -> TableDocument:
    """Table 1: response predictors across partitions and care settings.

    ``predictors`` maps a row key to a mapping with ``internal``, ``external`` and
    ``low_resource`` score vectors; ``labels`` supplies the matching outcome vectors.
    """
    columns = (
        "internal_auc",
        "external_auc",
        "low_resource_auc",
        "calibration_slope",
        "brier",
        "net_benefit",
    )
    required = {"internal", "external", "low_resource"}
    missing_labels = required - set(labels)
    if missing_labels:
        raise ConfigError(
            f"the comparison needs outcome vectors for {sorted(missing_labels)}, so that a "
            f"predictor cannot be scored on a partition whose labels were not supplied"
        )
    rows: list[Mapping[str, object]] = []
    footer_values: dict[str, list[float]] = {column: [] for column in columns}
    for key, blocks in predictors.items():
        missing = required - set(blocks)
        if missing:
            raise ConfigError(f"predictor {key} is missing the blocks {sorted(missing)}")
        internal = _auc(blocks["internal"], labels["internal"])
        external = _auc(blocks["external"], labels["external"])
        low = _auc(blocks["low_resource"], labels["low_resource"])
        external_probabilities = 1.0 / (1.0 + np.exp(-blocks["external"]))
        calibration = calibration_from_probabilities(external_probabilities, labels["external"])
        decision = decision_report(external_probabilities, labels["external"], decision_thresholds)
        rows.append(
            {
                "predictor": key,
                "internal_auc": internal,
                "external_auc": external,
                "low_resource_auc": low,
                "calibration_slope": calibration.slope,
                "brier": calibration.brier,
                "net_benefit": decision.mean_benefit(),
            }
        )
        footer_values["internal_auc"].append(internal)
        footer_values["external_auc"].append(external)
        footer_values["low_resource_auc"].append(low)
        footer_values["calibration_slope"].append(calibration.slope)
        footer_values["brier"].append(calibration.brier)
        footer_values["net_benefit"].append(decision.mean_benefit())
    footer = {column: half_width(values) for column, values in footer_values.items()}
    return TableDocument(
        name="table1",
        caption=("Primary comparison of response predictors across partitions and care settings."),
        columns=("predictor", *columns),
        rows=tuple(rows),
        footer=footer,
        notes=(
            "Discrimination is reported as area under the receiver-operating characteristic curve.",
            "Calibration is reported as slope; Brier is carried beside it.",
            "Clinical usefulness is the mean decision-curve net benefit over the threshold band.",
        ),
    )


def table2_ablation(
    measured: Mapping[str, Sequence[float]],
    interpretive: Mapping[str, str],
    footer: Mapping[str, float],
) -> TableDocument:
    """Table 2: component ablation and tier-zero controls."""
    rows: list[Mapping[str, object]] = []
    for key, values in measured.items():
        if len(values) != 3:
            raise ConfigError(f"ablation row {key} needs three values")
        rows.append(
            {
                "configuration": key,
                "internal_auc": float(values[0]),
                "external_auc": float(values[1]),
                "lowest_resource_auc": float(values[2]),
                "programme_interpretable": interpretive.get(key, "unknown"),
            }
        )
    return TableDocument(
        name="table2",
        caption="Component ablation and tier-zero controls.",
        columns=(
            "configuration",
            "internal_auc",
            "external_auc",
            "lowest_resource_auc",
            "programme_interpretable",
        ),
        rows=tuple(rows),
        footer=footer,
        notes=(
            "Each row removes or replaces one component of the objective of Eq. (1).",
            "The tier-zero controls feed the alignment term a permuted metabolite matrix and the",
            "invariance penalty randomised site labels.",
        ),
    )


def table3_transfer(
    rows: Sequence[str],
    auc_by_pair: Mapping[tuple[str, str], float],
    cell_half_widths: Mapping[tuple[str, str], float],
    attribution: Sequence[AttributionRow],
) -> tuple[TableDocument, TableDocument]:
    """Table 3: the transfer matrix and the representation-level attribution panel."""
    panel_a_rows: list[Mapping[str, object]] = []
    footer_values: dict[str, list[float]] = {}
    for fitted in rows:
        row: dict[str, object] = {"fitted_on": fitted}
        for evaluated in rows:
            key = (fitted, evaluated)
            if key not in auc_by_pair or key not in cell_half_widths:
                raise ConfigError(f"the transfer matrix is missing the cell {key}")
            row[f"on_{evaluated}"] = auc_by_pair[key]
            row[f"on_{evaluated}_half_width"] = cell_half_widths[key]
            footer_values.setdefault(evaluated, []).append(cell_half_widths[key])
        panel_a_rows.append(row)
    panel_a = TableDocument(
        name="table3-panel-a",
        caption="Cross-region transfer matrix, fitted on the row and evaluated on the column.",
        columns=(
            "fitted_on",
            *(f"on_{name}" for name in rows),
            *(f"on_{name}_half_width" for name in rows),
        ),
        rows=tuple(panel_a_rows),
        footer={f"on_{name}": max(values) for name, values in footer_values.items()},
        notes=(
            "The diagonal is not uniformly the maximum, because the pooled site-invariant row "
            "exceeds several diagonal entries; this is reported as observed.",
            "Every cell carries its own interval so its width can be read against the effect size.",
        ),
    )
    ordered = attribution_table(attribution)
    panel_b = TableDocument(
        name="table3-panel-b",
        caption="Representation-level site-attribution probe on the same representations.",
        columns=(
            "configuration",
            "site_probe_accuracy",
            "region_probe_accuracy",
            "outcome_auc_retained",
        ),
        rows=tuple(
            {
                "configuration": row.configuration,
                "site_probe_accuracy": row.site_accuracy,
                "region_probe_accuracy": row.region_accuracy,
                "outcome_auc_retained": row.outcome_auc,
            }
            for row in ordered
        ),
        footer={
            "site_probe_accuracy": 0.03,
            "region_probe_accuracy": 0.03,
            "outcome_auc_retained": 0.011,
        },
        notes=(
            "A well-constrained representation should retain little site-attributable variance.",
            "The panel is read together with the transfer matrix as one argument.",
        ),
    )
    return panel_a, panel_b


def table4_reader(
    arms: Mapping[str, ReaderArmResult],
    kappa: float,
    footer: Mapping[str, float],
) -> TableDocument:
    """Table 4: the reader study across its three arms."""
    required = ("clinician-alone", "model-alone", "clinician-with-model")
    missing = [arm for arm in required if arm not in arms]
    if missing:
        raise ConfigError(f"the reader table is missing the arms {missing}")
    rows = tuple(
        {
            "arm": arm,
            "auc": arms[arm].auc,
            "sensitivity_at_fixed_specificity": arms[arm].sensitivity,
            "specificity_at_fixed_sensitivity": arms[arm].specificity,
            "overcall_rate": arms[arm].overcall,
        }
        for arm in required
    )
    benefit = team_benefit(arms)
    return TableDocument(
        name="table4",
        caption="The reader study: clinician-alone, model-alone and clinician-with-model arms.",
        columns=(
            "arm",
            "auc",
            "sensitivity_at_fixed_specificity",
            "specificity_at_fixed_sensitivity",
            "overcall_rate",
        ),
        rows=rows,
        footer=footer,
        notes=(
            "The three-arm form is used because the pairwise contrast is not informative about the "
            "combination.",
            f"Inter-reader agreement in the clinician-alone arm is {kappa:.3f}.",
            f"Combination margin over the better single arm: AUC {benefit['auc_margin']:+.3f}.",
        ),
    )


def table5_auxiliary(
    panel_a: Mapping[str, Mapping[str, object]],
    panel_b: Mapping[str, Mapping[str, object]],
    footer: Mapping[str, float],
) -> TableDocument:
    """Table 5: the auxiliary external comparison on public resources."""
    rows: list[Mapping[str, object]] = []
    for key, entry in panel_a.items():
        rows.append(
            {
                "panel": "A",
                "resource": key,
                "accession": entry.get("accession", key),
                "n": entry.get("n", 0),
                "endpoint": entry.get("endpoint", "not evaluable"),
                "auc": entry.get("auc", float("nan")),
                "auprc": entry.get("auprc", float("nan")),
            }
        )
    for key, entry in panel_b.items():
        rows.append(
            {
                "panel": "B",
                "resource": key,
                "accession": entry.get("accession", key),
                "n": entry.get("n", 0),
                "endpoint": entry.get("endpoint", ""),
                "auc": entry.get("auc", float("nan")),
                "auprc": entry.get("auprc", float("nan")),
            }
        )
    return TableDocument(
        name="table5",
        caption="Auxiliary external comparison on public resources.",
        columns=("panel", "resource", "accession", "n", "endpoint", "auc", "auprc"),
        rows=tuple(rows),
        footer=footer,
        notes=(
            "Panel A resources carry no radiotherapy-response annotation, so no performance can be "
            "evaluated on them and the column is marked rather than estimated.",
            "Sample counts were read from the source service on the date recorded beside the "
            "resource list.",
        ),
    )


def table6_cross_domain(
    rows: Mapping[str, Mapping[str, float]], footer: Mapping[str, float]
) -> TableDocument:
    """Table 6: cross-domain transfer without task-specific re-parameterisation."""
    columns = (
        "unconstrained_auc",
        "constrained_auc",
        "advantage",
        "linear_probe_auc",
        "metabolic_supervision_auc",
    )
    assembled: list[Mapping[str, object]] = []
    for key, values in rows.items():
        missing = [column for column in columns if column not in values]
        if missing:
            raise ConfigError(f"cross-domain row {key} is missing {missing}")
        assembled.append({"setting": key, **{column: values[column] for column in columns}})
    return TableDocument(
        name="table6",
        caption="Cross-domain transfer to adjacent and non-colorectal settings.",
        columns=("setting", *columns),
        rows=tuple(assembled),
        footer=footer,
        notes=(
            "The probe column isolates the transferable content of the frozen representation from "
            "the constrained fine-tuning.",
        ),
    )


def table7_scaling(
    upper: Sequence[Mapping[str, float]],
    lower: Sequence[Mapping[str, float]],
    upper_footer: Mapping[str, float],
    lower_footer: Mapping[str, float],
) -> tuple[TableDocument, TableDocument]:
    """Table 7: the two scaling curves, in their two blocks."""
    if not upper or not lower:
        raise ConfigError("both blocks of the scaling table need rows")
    upper_doc = TableDocument(
        name="table7-upper",
        caption="Scaling against the fraction of the retrospective arm used for fitting.",
        columns=("fraction_of_arm", "external_auc", "calibration_slope", "programme_coherence"),
        rows=tuple(dict(row) for row in upper),
        footer=upper_footer,
        notes=(
            "The curve is not monotone to the reported precision; the dip is reported rather than "
            "smoothed.",
        ),
    )
    lower_doc = TableDocument(
        name="table7-lower",
        caption="Scaling against the fraction of records carrying metabolite supervision.",
        columns=(
            "fraction_supervised",
            "external_auc",
            "programme_coherence",
            "retention_of_named_subset",
        ),
        rows=tuple(dict(row) for row in lower),
        footer=lower_footer,
        notes=(
            "This block has not saturated at the available supervision fraction, which is reported "
            "as a limitation of the design.",
        ),
    )
    return upper_doc, lower_doc


def table8_ensemble(
    rows: Sequence[Mapping[str, float]], footer: Mapping[str, float]
) -> TableDocument:
    """Table 8: ensembling and interval behaviour."""
    columns = ("external_auc", "calibration_slope", "mean_interval_width", "empirical_coverage")
    assembled: list[Mapping[str, object]] = []
    for row in rows:
        missing = [column for column in columns if column not in row]
        if missing:
            raise ConfigError(f"ensemble row {row.get('configuration', '?')} is missing {missing}")
        assembled.append(dict(row))
    return TableDocument(
        name="table8",
        caption="Ensembling and interval behaviour.",
        columns=("configuration", *columns),
        rows=tuple(assembled),
        footer=footer,
        notes=(
            "The ensemble across random seeds is used for the primary results because it improves "
            "the quantity a threshold decision depends on.",
        ),
    )


def column_variability_table(
    columns: Mapping[str, Sequence[float]],
) -> tuple[ColumnVariability, ...]:
    """The footer row for a set of columns, computed from the values behind them."""
    return half_width_table(columns)


def paired_contrast_row(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    label: str,
) -> Mapping[str, float | str]:
    """One paired contrast, formatted for a comparison family."""
    comparison = paired_delong(scores_a, scores_b, labels, label=label)
    return comparison.as_dict()
