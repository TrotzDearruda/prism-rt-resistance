"""Training entry point.

Ref: Algorithm 1 -- one run fits the ensemble members through the two stages and writes the run
header before the first optimisation step, so a table can be traced back to the configuration and
the seed that produced it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence

import numpy as np

from prism.cli._shared import (
    add_cohort_argument,
    add_common_arguments,
    describe_config,
    entry,
    load_cohort,
)
from prism.fitting.ensemble import build_seed_ensemble
from prism.fitting.loop import evaluate_probabilities
from prism.fitting.stages import fit
from prism.numerics.seed import ensemble_seeds
from prism.provenance.journal import get_logger, record_run_header


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-train",
        description="Fit the constrained multimodal model on the cohort a run is configured for.",
    )
    add_common_arguments(parser)
    add_cohort_argument(parser)
    parser.add_argument(
        "--members",
        type=int,
        default=None,
        help="ensemble members to fit; defaults to the configured member count",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config, paths, device = entry(args, "cli.train")
    context = load_cohort(config, args.cohort)
    logger = get_logger("cli.train")
    n_members = config.train.ensemble_members if args.members is None else args.members
    if n_members < 1:
        raise SystemExit("an ensemble needs at least one member")

    record_run_header(
        paths.root,
        config.name,
        config,
        {layer.value: len(cohort) for layer, cohort in context.parts.items()},
        ensemble_seeds(n_members, base=config.train.seed),
    )
    logger.info("cohort source: %s", context.source)

    probabilities: list[np.ndarray] = []
    summaries: list[Mapping[str, object]] = []
    held_out = context.held_out()
    rows = np.arange(len(held_out), dtype=np.int64)
    for member in range(n_members):
        result = fit(config, context.parts, context.lexicon, device, paths, member=member)
        summaries.append(result.as_dict())
        if len(held_out) > 0:
            probabilities.append(
                evaluate_probabilities(
                    result.model, held_out, rows, config.train.batch_size, device
                )
            )
    if probabilities:
        ensemble = build_seed_ensemble(probabilities, config.train.seed)
        logger.info(
            "ensemble of %d members pools %d held-out rows",
            ensemble.n_members(),
            ensemble.pooled().shape[0],
        )
    (paths.root / "fit_summary.json").write_text(
        json.dumps(
            {"config": describe_config(config), "members": summaries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
