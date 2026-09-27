"""Cohort records and the leakage boundary between observation and latent truth.

Ref: Methods Sec. 6.1 defines the observation X, the response Y, the care setting S and
the metabolite vector M. Ref: Methods Sec. 6.3 -- M is a training-time supervisor and is
discarded at inference, so a record exposes its metabolite block through one accessor
that the inference path never calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from prism.errors import SupervisionError
from prism.notation import MODALITY_STREAMS


class Layer(StrEnum):
    """Which partition a record belongs to.

    Ref: Results Sec. 2.1 -- the internal partition is separated from the development arm
    at the patient level; the external partition is the silent exposure sample.
    """

    DEVELOPMENT = "development"
    INTERNAL = "internal"
    EXTERNAL = "external"
    SILENT = "silent"


INFERENCE_LAYERS: tuple[Layer, ...] = (Layer.INTERNAL, Layer.EXTERNAL, Layer.SILENT)


@dataclass(frozen=True)
class CohortRecord:
    """One observation, addressed both by its setting and by its available modalities."""

    record_id: str
    patient_id: str
    site: str
    region: str
    stratum: str
    outcome: int
    layer: Layer
    features: Mapping[str, np.ndarray]
    availability: tuple[str, ...]
    metabolite: np.ndarray | None = None

    def stream(self, name: str) -> np.ndarray:
        try:
            return self.features[name]
        except KeyError as failure:
            raise SupervisionError(f"record {self.record_id} carries no {name} stream") from failure

    def has(self, name: str) -> bool:
        return name in self.features and name in self.availability

    def supervised(self) -> bool:
        return self.metabolite is not None

    def supervision(self) -> np.ndarray:
        """The metabolite block. Only the fitting path may call this."""
        if self.metabolite is None:
            raise SupervisionError(
                f"record {self.record_id} carries no spatially resolved metabolite block"
            )
        return self.metabolite

    def inference_visible(self) -> Mapping[str, np.ndarray]:
        """Streams the deployed model may read.

        Ref: Methods Sec. 6.3 -- the metabolic modality is never available at inference.
        """
        return {
            name: block
            for name, block in self.features.items()
            if name in self.availability and name in MODALITY_STREAMS
        }


@dataclass(frozen=True)
class Cohort:
    """An immutable, patient-indexed bundle of records."""

    records: tuple[CohortRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CohortRecord]:
        return iter(self.records)

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({record.site for record in self.records}))

    def regions(self) -> tuple[str, ...]:
        return tuple(sorted({record.region for record in self.records}))

    def strata(self) -> tuple[str, ...]:
        return tuple(sorted({record.stratum for record in self.records}))

    def patient_ids(self) -> tuple[str, ...]:
        return tuple(record.patient_id for record in self.records)

    def patient_keys(self) -> tuple[str, ...]:
        """The resampling unit.

        Ref: Methods Sec. 6.10 -- uncertainty is drawn over patients, and records of one
        patient always move together.
        """
        return tuple(record.patient_id for record in self.records)

    def labels(self) -> np.ndarray:
        return np.asarray([record.outcome for record in self.records], dtype=np.int64)

    def layers(self) -> np.ndarray:
        return np.asarray([record.layer.value for record in self.records], dtype=object)

    def site_labels(self) -> np.ndarray:
        return np.asarray([record.site for record in self.records], dtype=object)

    def region_labels(self) -> np.ndarray:
        return np.asarray([record.region for record in self.records], dtype=object)

    def stratum_labels(self) -> np.ndarray:
        return np.asarray([record.stratum for record in self.records], dtype=object)

    def layer_counts(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.layer.value] = counts.get(record.layer.value, 0) + 1
        return counts

    def subset(self, layer: Layer) -> Cohort:
        return Cohort(tuple(record for record in self.records if record.layer is layer))

    def restrict(self, keep: Callable[[CohortRecord], bool]) -> Cohort:
        return Cohort(tuple(record for record in self.records if keep(record)))

    def by_site(self, site: str) -> Cohort:
        return Cohort(tuple(record for record in self.records if record.site == site))

    def by_region(self, region: str) -> Cohort:
        return Cohort(tuple(record for record in self.records if record.region == region))

    def by_stratum(self, stratum: str) -> Cohort:
        return Cohort(tuple(record for record in self.records if record.stratum == stratum))

    def supervised(self) -> Cohort:
        return Cohort(tuple(record for record in self.records if record.supervised()))

    def stream_matrix(self, name: str) -> np.ndarray:
        """Stack one modality into a row-major matrix, zero-filling absent rows."""
        columns: int | None = None
        blocks: list[np.ndarray] = []
        for record in self.records:
            if record.has(name):
                block = np.asarray(record.stream(name), dtype=np.float64).reshape(-1)
            elif columns is None:
                reference = self._first_shape(name)
                block = np.zeros(reference, dtype=np.float64)
            else:
                block = np.zeros(columns, dtype=np.float64)
            columns = block.shape[0]
            blocks.append(block)
        if not blocks:
            return np.zeros((0, 0), dtype=np.float64)
        return np.stack(blocks, axis=0)

    def _first_shape(self, name: str) -> int:
        for record in self.records:
            if name in record.features:
                return int(np.asarray(record.features[name]).size)
        raise SupervisionError(f"no record carries a {name} stream to size the matrix")

    def availability_matrix(self, streams: Sequence[str] = MODALITY_STREAMS) -> np.ndarray:
        """Boolean indicator of which modality each site could supply.

        Ref: Results Sec. 2.6 -- the resource stratum changes how many modalities a site
        has, and the modality dropout scheme is the mechanism for reading that.
        """
        mask = np.zeros((len(self.records), len(streams)), dtype=bool)
        for row, record in enumerate(self.records):
            for column, name in enumerate(streams):
                mask[row, column] = record.has(name)
        return mask

    def metabolite_matrix(self) -> np.ndarray:
        """Stack the supervised metabolite blocks, one row per supervised record."""
        rows = [
            np.asarray(record.supervision(), dtype=np.float64).reshape(-1)
            for record in self.records
            if record.supervised()
        ]
        if not rows:
            return np.zeros((0, 0), dtype=np.float64)
        return np.stack(rows, axis=0)

    def supervision_rows(self) -> np.ndarray:
        return np.asarray(
            [position for position, record in enumerate(self.records) if record.supervised()],
            dtype=np.int64,
        )
