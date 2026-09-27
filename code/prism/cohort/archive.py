"""Staged-cohort archive.

The clinical arm of the study is held by the participating institutions and is not redistributable,
so this release ships no way to manufacture a cohort. It ships the boundary instead: a run reads the
arm the data owner staged as one archive, and reports that the arm is absent when it is not there.

The archive is a single ``.npz`` file holding rectangular arrays. Every stream is stacked, so a
record whose site could not acquire a modality carries zeros there and the availability matrix marks
the row; the metabolite block is stacked the same way with the supervision flag beside it. The
modality names and their shapes are written into the archive, so a reader can refuse an archive that
does not match the configuration it is being read with rather than mis-index it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prism.cohort.schema import Cohort, CohortRecord, Layer
from prism.errors import CohortError
from prism.notation import MODALITY_STREAMS

SCHEMA_VERSION = 1

RECORD_FIELDS = (
    "record_id",
    "patient_id",
    "site",
    "region",
    "stratum",
    "outcome",
    "layer",
)


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


@dataclass(frozen=True)
class ArchiveContents:
    """What an archive declares about itself before a record is read."""

    schema_version: int
    modality_names: tuple[str, ...]
    stream_shapes: Mapping[str, tuple[int, ...]]
    metabolite_width: int
    n_records: int
    layer_counts: Mapping[str, int]
    provenance: str

    def validate(self, expected_shapes: Mapping[str, tuple[int, ...]] | None) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CohortError(
                f"the archive declares schema {self.schema_version}, this release reads "
                f"{SCHEMA_VERSION}"
            )
        if self.n_records < 1:
            raise CohortError("the archive carries no record")
        if self.metabolite_width < 1:
            raise CohortError("the archive declares no metabolite vocabulary")
        if expected_shapes is None:
            return
        for name, shape in expected_shapes.items():
            if name not in self.stream_shapes:
                raise CohortError(f"the archive carries no {name} stream")
            if tuple(self.stream_shapes[name]) != tuple(shape):
                raise CohortError(
                    f"the archive carries {name} at shape {tuple(self.stream_shapes[name])} while "
                    f"the configuration declares {tuple(shape)}"
                )


def write_archive(
    cohort: Cohort,
    path: Path,
    *,
    metabolite_width: int | None = None,
    provenance: str = "",
) -> ArchiveContents:
    """Write a cohort as one archive, recording the shapes it was written with."""
    if len(cohort) == 0:
        raise CohortError("an empty cohort cannot be archived")
    names = tuple(
        name
        for name in MODALITY_STREAMS
        if any(name in record.features for record in cohort.records)
    )
    if not names:
        raise CohortError("the cohort carries no stream that this release can read")
    shapes: dict[str, tuple[int, ...]] = {}
    for name in names:
        for record in cohort.records:
            block = record.features.get(name)
            if block is not None:
                shapes[name] = tuple(int(dimension) for dimension in np.asarray(block).shape)
                break

    payload: dict[str, Any] = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "modality_names": _string_array(names),
        "provenance": np.asarray(provenance, dtype=np.str_),
    }
    for name in names:
        payload[f"shape__{name}"] = np.asarray(shapes[name], dtype=np.int64)
        rows = [
            (
                np.asarray(record.features[name], dtype=np.float64)
                if record.has(name)
                else np.zeros(shapes[name], dtype=np.float64)
            )
            for record in cohort.records
        ]
        payload[f"stream__{name}"] = np.stack(rows, axis=0)
    payload["availability"] = np.asarray(
        [[record.has(name) for name in MODALITY_STREAMS] for record in cohort.records],
        dtype=bool,
    )

    metabolite_rows: list[np.ndarray] = []
    width = metabolite_width or 0
    for record in cohort.records:
        if record.supervised():
            block = np.asarray(record.supervision(), dtype=np.float64).reshape(-1)
            width = int(block.size)
            metabolite_rows.append(block)
        else:
            metabolite_rows.append(np.zeros(width, dtype=np.float64))
    if width < 1:
        raise CohortError(
            "the archive needs a metabolite width, which comes from the supervised records or from "
            "the argument"
        )
    payload["metabolite"] = np.stack(
        [
            row if row.size == width else np.zeros(width, dtype=np.float64)
            for row in metabolite_rows
        ],
        axis=0,
    )
    payload["supervised"] = np.asarray(
        [record.supervised() for record in cohort.records], dtype=bool
    )

    for field in RECORD_FIELDS:
        values = [getattr(record, field) for record in cohort.records]
        if field == "outcome":
            payload[field] = np.asarray(values, dtype=np.int64)
        else:
            payload[field] = _string_array(values)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return describe_archive(path)


def describe_archive(path: Path) -> ArchiveContents:
    """Read what an archive declares about itself, without building the records."""
    with _open(path) as handle:
        names = tuple(str(name) for name in handle["modality_names"])
        shapes = {name: tuple(int(value) for value in handle[f"shape__{name}"]) for name in names}
        layers = np.asarray(handle["layer"], dtype=object)
        counted: dict[str, int] = {}
        for value in layers.tolist():
            counted[str(value)] = counted.get(str(value), 0) + 1
        return ArchiveContents(
            schema_version=int(handle["schema_version"]),
            modality_names=names,
            stream_shapes=shapes,
            metabolite_width=int(np.asarray(handle["metabolite"]).shape[1]),
            n_records=int(np.asarray(handle["outcome"]).shape[0]),
            layer_counts=counted,
            provenance=str(handle["provenance"]),
        )


def read_archive(
    path: Path, *, expected_shapes: Mapping[str, tuple[int, ...]] | None = None
) -> Cohort:
    """Read a staged cohort, refusing an archive that does not match the configuration."""
    contents = describe_archive(path)
    contents.validate(expected_shapes)
    with _open(path) as handle:
        outcomes = np.asarray(handle["outcome"], dtype=np.int64)
        availability = np.asarray(handle["availability"], dtype=bool)
        metabolite = np.asarray(handle["metabolite"], dtype=np.float64)
        supervised = np.asarray(handle["supervised"], dtype=bool)
        ids = np.asarray(handle["record_id"], dtype=object)
        patients = np.asarray(handle["patient_id"], dtype=object)
        sites = np.asarray(handle["site"], dtype=object)
        regions = np.asarray(handle["region"], dtype=object)
        strata = np.asarray(handle["stratum"], dtype=object)
        layers = np.asarray(handle["layer"], dtype=object)
        streams = {
            name: np.asarray(handle[f"stream__{name}"], dtype=np.float64)
            for name in contents.modality_names
        }

        records: list[CohortRecord] = []
        for row in range(contents.n_records):
            available = tuple(
                name
                for column, name in enumerate(MODALITY_STREAMS)
                if bool(availability[row, column])
            )
            features = {name: streams[name][row] for name in available if name in streams}
            records.append(
                CohortRecord(
                    record_id=str(ids[row]),
                    patient_id=str(patients[row]),
                    site=str(sites[row]),
                    region=str(regions[row]),
                    stratum=str(strata[row]),
                    outcome=int(outcomes[row]),
                    layer=_layer(str(layers[row])),
                    features=features,
                    availability=available,
                    metabolite=metabolite[row] if bool(supervised[row]) else None,
                )
            )
    return Cohort(tuple(records))


def _layer(name: str) -> Layer:
    for candidate in Layer:
        if candidate.value == name:
            return candidate
    raise CohortError(
        f"the archive declares a partition named {name!r}, which this release does not"
    )


class _Archive:
    """Context manager over the loaded archive file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> Any:
        if not self.path.is_file():
            raise CohortError(
                f"no cohort is staged at {self.path}; the clinical arm of the study is held by the "
                f"participating institutions and must be staged before a run can read it"
            )
        self.handle = np.load(self.path, allow_pickle=False)
        return self.handle

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            self.handle.close()


def _open(path: Path) -> _Archive:
    return _Archive(path)


def default_archive_path(root: Path) -> Path:
    """Where a staged arm is expected to sit under a data root."""
    return root / "cohort.npz"
