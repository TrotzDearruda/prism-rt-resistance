"""Evaluation entry point.

Ref: Results Sec. 2 -- the evaluation reads the held-out partitions, which are never touched inside
the fitting loop. A member is fitted, its scores are read on the internal, external and
lowest-resource views of Table 1, the representation-level attribution panel is read on the
internal view, and the cross-region matrix is filled by fitting a closed-form probe per region on
those same representations.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np
import torch

from prism.assessment.harness import table1_predictors, table3_transfer
from prism.assessment.reporting import TableDocument, write_bundle
from prism.cli._shared import (
    add_cohort_argument,
    add_common_arguments,
    as_float,
    entry,
    load_cohort,
)
from prism.cohort.views import INTERNAL, partition_views
from prism.fitting.loop import evaluate_logits, evaluate_representations
from prism.fitting.stages import fit
from prism.measurement.attribution import AttributionRow, attribution_row
from prism.measurement.discrimination import rank_auc
from prism.modelling.probe import fit_probe_closed_form, probe_scores
from prism.provenance.journal import get_logger

TABLE3_FOOTER = {
    "site_probe_accuracy": 0.03,
    "region_probe_accuracy": 0.03,
    "outcome_auc_retained": 0.011,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-evaluate",
        description="Read the held-out partitions and write the reported tables.",
    )
    add_common_arguments(parser)
    add_cohort_argument(parser)
    parser.add_argument("--member", type=int, default=0, help="ensemble member to evaluate")
    return parser


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def cross_region_matrix(
    representations: np.ndarray,
    regions: np.ndarray,
    outcomes: np.ndarray,
    *,
    ridge: float = 1e-2,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], tuple[str, ...]]:
    """Fill a transfer matrix by fitting a closed-form probe on each region.

    Ref: Table 3 Panel A -- a model fitted on the region named by the row is evaluated on the region
    named by the column. The probe is fitted in closed form so the matrix does not depend on an
    optimisation schedule, and a region whose probe cannot be fitted is dropped rather than filled
    with a guess.
    """
    labels = np.asarray([str(region) for region in np.asarray(regions, dtype=object)], dtype=object)
    truth = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    if labels.size != truth.size:
        raise ValueError("region labels and outcomes must share a length")
    names = tuple(sorted(set(labels.tolist())))
    aucs: dict[tuple[str, str], float] = {}
    widths: dict[tuple[str, str], float] = {}
    for fitted_on in names:
        train_mask = labels == fitted_on
        if int(train_mask.sum()) < 8:
            continue
        train_labels = truth[train_mask]
        if np.unique(train_labels).size < 2:
            continue
        fit_result = fit_probe_closed_form(
            _as_tensor(representations[train_mask]), torch.from_numpy(train_labels), ridge=ridge
        )
        for evaluated_on in names:
            test_mask = labels == evaluated_on
            if int(test_mask.sum()) < 4:
                continue
            test_labels = truth[test_mask]
            if np.unique(test_labels).size < 2:
                continue
            scores = probe_scores(_as_tensor(representations[test_mask]), fit_result).numpy()
            aucs[(fitted_on, evaluated_on)] = rank_auc(scores, test_labels)
            widths[(fitted_on, evaluated_on)] = 1.0 / float(np.sqrt(int(test_mask.sum())))
    kept = tuple(name for name in names if any(key[0] == name for key in aucs))
    return aucs, widths, kept


def _attribution_document(attribution: AttributionRow) -> TableDocument:
    row = attribution
    return TableDocument(
        name="table3-panel-b",
        caption="Representation-level site-attribution probe on the same representations.",
        columns=(
            "configuration",
            "site_probe_accuracy",
            "region_probe_accuracy",
            "outcome_auc_retained",
        ),
        rows=(
            {
                "configuration": str(row.configuration),
                "site_probe_accuracy": float(row.site_accuracy),
                "region_probe_accuracy": float(row.region_accuracy),
                "outcome_auc_retained": float(row.outcome_auc),
            },
        ),
        footer=TABLE3_FOOTER,
        notes=(
            "No region carried enough rows on both sides of a transfer to fill the matrix; the "
            "attribution panel is written on its own.",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config, paths, device = entry(args, "cli.evaluate")
    context = load_cohort(config, args.cohort)
    logger = get_logger("cli.evaluate")
    result = fit(config, context.parts, context.lexicon, device, paths, member=args.member)

    views = partition_views(context.parts)
    scores: dict[str, dict[str, np.ndarray]] = {"prism_constrained": {}}
    labels: dict[str, np.ndarray] = {}
    for name, view in views.items():
        scores["prism_constrained"][name] = evaluate_logits(
            result.model, view.cohort, view.rows(), config.train.batch_size, device
        )
        labels[name] = view.labels
    table1 = table1_predictors(scores, labels, (0.10, 0.15, 0.20, 0.25, 0.30))

    inner = views[INTERNAL]
    representations = evaluate_representations(
        result.model, inner.cohort, inner.rows(), config.train.batch_size, device
    )
    attribution = attribution_row(
        "constrained_variance_penalty",
        representations,
        inner.cohort.site_labels(),
        inner.cohort.region_labels(),
        as_float(table1.rows[0]["internal_auc"], "table1.internal_auc"),
    )
    aucs, widths, kept = cross_region_matrix(
        representations,
        np.asarray(inner.cohort.region_labels(), dtype=object),
        inner.labels,
    )
    if kept:
        panel_a, panel_b = table3_transfer(
            rows=kept,
            auc_by_pair=aucs,
            cell_half_widths=widths,
            attribution=(attribution,),
        )
        documents: tuple[TableDocument, ...] = (table1, panel_a, panel_b)
    else:
        documents = (table1, _attribution_document(attribution))
    digests = write_bundle(paths.tables, documents)
    (paths.root / "evaluation_summary.json").write_text(
        json.dumps(
            {"tables": sorted(digests), "digests": digests, "regions": list(kept)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %d table documents over %d regions", len(digests), len(kept))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
