"""Programme emission entry point.

Ref: Methods Sec. 6.1 -- the reported output is a programme with signs and a per-programme effect
next to its interval, not a risk score. This entry point reads the held-out partitions, emits the
programmes, and writes them next to a summary of support size, sign agreement and programme
ranking.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np
import torch

from prism.cli._shared import (
    add_cohort_argument,
    add_common_arguments,
    entry,
    load_cohort,
)
from prism.cohort.views import EXTERNAL, INTERNAL, partition_views
from prism.emission.programmes import (
    effect_table,
    emission_diagnostics,
    emit_programmes,
    programme_ranking,
    sign_agreement_of_statements,
    statement_table,
    support_summary,
)
from prism.fitting.loop import evaluate_coefficients, evaluate_representations
from prism.fitting.stages import fit
from prism.modelling.interval_head import interval_from_scale
from prism.provenance.journal import get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-emit",
        description="Emit the signed metabolite programme and its per-programme effects.",
    )
    add_common_arguments(parser)
    add_cohort_argument(parser)
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument(
        "--partition", default=EXTERNAL, choices=(INTERNAL, EXTERNAL), help="view to emit for"
    )
    return parser


def interval_columns(
    beta: torch.Tensor, scale: torch.Tensor, nominal_coverage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Effect and interval bounds from the interval head output."""
    lower, upper = interval_from_scale(beta, scale, nominal_coverage)
    return lower.numpy(), upper.numpy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config, paths, device = entry(args, "cli.emit")
    context = load_cohort(config, args.cohort)
    result = fit(config, context.parts, context.lexicon, device, paths, member=args.member)
    views = partition_views(context.parts)
    view = views[args.partition]
    logger = get_logger("cli.emit")

    model = result.model
    model.eval()
    with torch.no_grad():
        coefficients = evaluate_coefficients(
            model, view.cohort, view.rows(), config.train.batch_size, device
        )
        representations = evaluate_representations(
            model, view.cohort, view.rows(), config.train.batch_size, device
        )
        interval_head = model.interval
        if interval_head is None:
            beta = np.zeros((len(view), context.lexicon.n_blocks()), dtype=np.float64)
            scale = np.ones_like(beta)
        else:
            effect, spread = interval_head(torch.from_numpy(representations.astype(np.float32)))
            beta = effect.numpy().astype(np.float64)
            scale = spread.numpy().astype(np.float64)
    expected = {entry.name: entry.expected_sign for entry in context.lexicon.catalogue()}
    statements = emit_programmes(
        coefficients,
        tuple(record.record_id for record in view.cohort.records),
        context.lexicon,
        beta,
        scale,
        nominal_coverage=config.train.interval_nominal_coverage,
    )
    payload = {
        "partition": view.name,
        "nominal_coverage": config.train.interval_nominal_coverage,
        "support": dict(support_summary(statements)),
        "sign_agreement": sign_agreement_of_statements(statements, context.lexicon),
        "ranking": [list(item) for item in programme_ranking(statements)],
        "effects": {
            key: dict(value)
            for key, value in effect_table(statements, context.lexicon.programme_ids).items()
        },
        "diagnostics": dict(emission_diagnostics(coefficients, context.lexicon, expected)),
        "statements": list(statement_table(statements)),
    }
    target = paths.emissions / f"{view.name}_programmes.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("emitted %d programmes to %s", len(statements), target.name)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
