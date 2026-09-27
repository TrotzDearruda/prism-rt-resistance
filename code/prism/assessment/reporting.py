"""Table documents and their serialisation.

Ref: Methods Sec. 6.10 -- every table carries the numerical variability of its estimates: the
final row gives the mean prediction-interval half-width per column, the cells of the transfer
matrix carry their own interval, and the notes repeat the half-widths so that an effect can be
sized against its interval from the table alone. A document therefore holds columns, rows, a
footer row and notes, and a table written without a footer is refused rather than shipped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from prism.errors import ConfigError
from prism.provenance.digest import digest_bytes

DEFAULT_LEVEL = 0.90


@dataclass(frozen=True)
class TableDocument:
    """One reported table."""

    name: str
    caption: str
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]
    footer: Mapping[str, float]
    coverage_level: float = DEFAULT_LEVEL
    notes: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.columns:
            raise ConfigError(f"table {self.name} declares no columns")
        if not self.rows:
            raise ConfigError(f"table {self.name} carries no rows")
        for row in self.rows:
            missing = [column for column in self.columns if column not in row]
            if missing:
                raise ConfigError(f"table {self.name} has a row missing the columns {missing}")
        unknown = [column for column in self.footer if column not in self.columns]
        if unknown:
            raise ConfigError(
                f"table {self.name} reports a footer for columns it does not carry: {unknown}"
            )
        if not self.footer:
            raise ConfigError(
                f"table {self.name} carries no interval footer; the study reports variability "
                f"with every table"
            )

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "name": self.name,
            "caption": self.caption,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "footer": {key: float(value) for key, value in self.footer.items()},
            "coverage_level": self.coverage_level,
            "notes": list(self.notes),
        }


def footer_note(footer: Mapping[str, float], level: float) -> str:
    """The sentence that repeats the half-widths inside the table notes."""
    parts = ", ".join(f"{column} {value:.3f}" for column, value in sorted(footer.items()))
    return f"{int(round(level * 100))}% prediction-interval half-width, by column: {parts}."


def with_footer_note(document: TableDocument) -> TableDocument:
    """Append the footer sentence to a document's notes if it is not already there."""
    sentence = footer_note(document.footer, document.coverage_level)
    if sentence in document.notes:
        return document
    return TableDocument(
        name=document.name,
        caption=document.caption,
        columns=document.columns,
        rows=document.rows,
        footer=document.footer,
        coverage_level=document.coverage_level,
        notes=(*document.notes, sentence),
    )


def write_document(out_dir: Path, document: TableDocument) -> Path:
    """Write one table document next to its peers."""
    prepared = with_footer_note(document)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{prepared.name}.json"
    body = json.dumps(prepared.as_dict(), indent=2, sort_keys=True)
    target.write_text(body, encoding="utf-8")
    return target


def write_bundle(out_dir: Path, documents: Sequence[TableDocument]) -> Mapping[str, str]:
    """Write every document of a run and return their digests."""
    digests: dict[str, str] = {}
    for document in documents:
        target = write_document(out_dir, document)
        digests[document.name] = digest_bytes(target.read_bytes())
    return digests


def compare_to_reference(
    measured: Mapping[str, float], reference: Mapping[str, float], tolerance: float
) -> Mapping[str, Mapping[str, float | str]]:
    """Read measured values against transcribed ones cell by cell.

    A study on a held-out cohort that is not redistributable cannot produce the same numbers as the
    article, so the comparison reports the gap and whether it falls inside the stated tolerance
    rather than declaring agreement or disagreement.
    """
    comparison: dict[str, Mapping[str, float | str]] = {}
    for key, value in measured.items():
        if key not in reference:
            comparison[key] = {
                "measured": float(value),
                "reference": float("nan"),
                "status": "no reference cell",
            }
            continue
        gap = float(value) - float(reference[key])
        comparison[key] = {
            "measured": float(value),
            "reference": float(reference[key]),
            "gap": gap,
            "within_tolerance": abs(gap) <= tolerance,
        }
    return comparison


def note_measurement_origin(document: TableDocument, origin: str) -> TableDocument:
    """Attach a note stating what population a table's values were measured on."""
    if not origin:
        raise ConfigError("a measurement origin note cannot be empty")
    return TableDocument(
        name=document.name,
        caption=document.caption,
        columns=document.columns,
        rows=document.rows,
        footer=document.footer,
        coverage_level=document.coverage_level,
        notes=(*document.notes, origin),
    )


def summarise_documents(documents: Sequence[TableDocument]) -> Mapping[str, object]:
    """Counts over a bundle, written to the run header."""
    names = [document.name for document in documents]
    if len(set(names)) != len(names):
        raise ConfigError("two documents in a bundle share a name")
    return {
        "n_documents": len(documents),
        "names": names,
        "n_rows": int(sum(len(document.rows) for document in documents)),
        "coverage_levels": sorted({document.coverage_level for document in documents}),
    }
