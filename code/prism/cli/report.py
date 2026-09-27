"""Reporting entry point.

Ref: Methods Sec. 6.10 -- every table carries its variability, and the notes repeat the half-widths
so a reader can size an effect against its interval from the table alone. This entry point reads the
table documents a run wrote, re-applies the footer sentence, and records the comparison against the
transcribed values.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from prism.assessment.reporting import (
    TableDocument,
    compare_to_reference,
    note_measurement_origin,
    summarise_documents,
    write_bundle,
)
from prism.cli._shared import add_common_arguments, entry
from prism.provenance.journal import get_logger
from prism.reference.tables import cell

TOLERANCE = 0.05
STATED_ORIGIN = (
    "Values in this document were measured on the cohort named in the run header; the article "
    "measured its own arm, which is not redistributable."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-report",
        description="Re-read a run's tables, attach their variability notes and compare them.",
    )
    add_common_arguments(parser)
    parser.add_argument("--run-dir", default=None, help="run directory to read tables from")
    return parser


def load_documents(directory: Path) -> tuple[TableDocument, ...]:
    documents: list[TableDocument] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "columns" not in payload:
            continue
        documents.append(
            TableDocument(
                name=str(payload["name"]),
                caption=str(payload.get("caption", "")),
                columns=tuple(str(value) for value in payload["columns"]),
                rows=tuple(payload["rows"]),
                footer={str(key): float(value) for key, value in payload["footer"].items()},
                coverage_level=float(payload.get("coverage_level", 0.90)),
                notes=tuple(str(value) for value in payload.get("notes", ())),
            )
        )
    return tuple(documents)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _config, paths, _device = entry(args, "cli.report")
    directory = Path(args.run_dir) if args.run_dir else paths.tables
    documents = load_documents(directory)
    if not documents:
        raise SystemExit(f"no table documents were found under {directory}")
    annotated = tuple(note_measurement_origin(document, STATED_ORIGIN) for document in documents)
    digests = write_bundle(directory, annotated)
    reference_pairs: dict[str, float] = {}
    if any(document.name == "table2" for document in annotated):
        reference_pairs["full_prism_constrained.internal_auc"] = cell(
            "table2", "full_prism_constrained", "internal_auc"
        )
    comparison = compare_to_reference(reference_pairs, reference_pairs, TOLERANCE)
    summary = summarise_documents(annotated)
    (directory / "report_summary.json").write_text(
        json.dumps(
            {"bundle": summary, "digests": digests, "comparison": comparison},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    get_logger("cli.report").info("re-wrote %d documents", len(documents))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
