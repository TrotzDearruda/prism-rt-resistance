"""Tensorisation of cohort records.

Ref: Methods Sec. 6.3 -- the observation carries a volume, a histopathology block and
laboratory values, and the fusion layer has to be told which streams a site could supply.
This module is the only place where a record becomes a tensor, so the availability rule of
Results Sec. 2.6 is applied once rather than at every call site.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor

from prism.cohort.schema import Cohort, CohortRecord
from prism.errors import SupervisionError
from prism.modelling.backbone import TensorBatch
from prism.notation import MODALITY_STREAMS

IndexSet: TypeAlias = Sequence[int] | np.ndarray


def _stream_rank(records: Sequence[CohortRecord], name: str) -> int | None:
    """Dimensionality of a stream as it appears on disk, or ``None`` if absent."""
    for record in records:
        block = record.features.get(name)
        if block is not None:
            return int(np.asarray(block).ndim)
    return None


def stack_volume(records: Sequence[CohortRecord], name: str) -> Tensor:
    """Stack a volumetric stream, zero-filling records whose site cannot acquire it."""
    shapes = [
        np.asarray(record.features[name]).shape for record in records if name in record.features
    ]
    if not shapes:
        raise SupervisionError(f"no record in the batch carries a {name} stream")
    shape = shapes[0]
    rows: list[np.ndarray] = []
    for record in records:
        if record.has(name):
            rows.append(np.asarray(record.features[name], dtype=np.float32))
        else:
            rows.append(np.zeros(shape, dtype=np.float32))
    return torch.from_numpy(np.stack(rows, axis=0))


def stack_vector(records: Sequence[CohortRecord], name: str) -> Tensor:
    """Stack a one-dimensional stream, zero-filling records that cannot acquire it."""
    width: int | None = None
    for record in records:
        block = record.features.get(name)
        if block is not None and np.asarray(block).ndim == 1:
            width = int(np.asarray(block).size)
            break
    if width is None:
        raise SupervisionError(f"no record in the batch carries a {name} stream")
    rows: list[np.ndarray] = []
    for record in records:
        if record.has(name):
            rows.append(np.asarray(record.features[name], dtype=np.float32).reshape(-1))
        else:
            rows.append(np.zeros(width, dtype=np.float32))
    return torch.from_numpy(np.stack(rows, axis=0))


def stack_bag(records: Sequence[CohortRecord], name: str) -> tuple[Tensor, Tensor]:
    """Pad a set-valued stream to the largest block in the batch and return its mask."""
    present = [
        np.asarray(record.features[name], dtype=np.float32)
        for record in records
        if record.has(name)
    ]
    if not present:
        raise SupervisionError(f"no record in the batch carries a {name} stream")
    for block in present:
        if block.ndim != 2:
            raise SupervisionError(f"{name} padding expects two-dimensional blocks")
    patch_dim = int(present[0].shape[1])
    widest = max(int(block.shape[0]) for block in present)
    n_rows = len(records)
    padded = np.zeros((n_rows, widest, patch_dim), dtype=np.float32)
    mask = np.zeros((n_rows, widest), dtype=bool)
    for row, record in enumerate(records):
        if not record.has(name):
            continue
        block = np.asarray(record.features[name], dtype=np.float32)
        count = int(block.shape[0])
        padded[row, :count, :] = block
        mask[row, :count] = True
    return torch.from_numpy(padded), torch.from_numpy(mask)


def build_batch(cohort: Cohort, indices: IndexSet) -> TensorBatch:
    """Turn a set of row positions into the tensor batch the backbone reads."""
    records = [cohort.records[int(index)] for index in indices]
    if not records:
        raise SupervisionError("an empty batch cannot be tensorised")
    streams: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    for name in MODALITY_STREAMS:
        rank = _stream_rank(records, name)
        if rank is None:
            continue
        if rank == 4:
            streams[name] = stack_volume(records, name)
        elif rank == 2:
            padded, mask = stack_bag(records, name)
            streams[name] = padded
            masks[name] = mask
        elif rank == 1:
            streams[name] = stack_vector(records, name)
        else:
            raise SupervisionError(f"{name} carries blocks of rank {rank}, which no encoder reads")
    availability = batch_availability(cohort, indices)
    return TensorBatch(
        streams=streams,
        present=torch.from_numpy(availability),
        masks=masks,
        record_ids=tuple(record.record_id for record in records),
    )


def labels_for(cohort: Cohort, indices: IndexSet) -> Tensor:
    values = np.asarray([cohort.records[int(index)].outcome for index in indices], dtype=np.float32)
    return torch.from_numpy(values)


def site_index_for(cohort: Cohort, indices: IndexSet) -> Tensor:
    """Site positions, ordered by the cohort's sorted site list so folds are stable."""
    sites = cohort.sites()
    lookup = {site: position for position, site in enumerate(sites)}
    values = np.asarray(
        [lookup[cohort.records[int(index)].site] for index in indices], dtype=np.int64
    )
    return torch.from_numpy(values)


def region_index_for(cohort: Cohort, indices: IndexSet) -> Tensor:
    regions = cohort.regions()
    lookup = {region: position for position, region in enumerate(regions)}
    values = np.asarray(
        [lookup[cohort.records[int(index)].region] for index in indices], dtype=np.int64
    )
    return torch.from_numpy(values)


def metabolite_for(cohort: Cohort, indices: IndexSet) -> Tensor:
    """Metabolite blocks for the supervised rows of a batch."""
    rows: list[np.ndarray] = []
    for index in indices:
        record = cohort.records[int(index)]
        if record.supervised():
            rows.append(np.asarray(record.supervision(), dtype=np.float32).reshape(-1))
    if not rows:
        raise SupervisionError("the batch carries no spatially resolved supervision")
    return torch.from_numpy(np.stack(rows, axis=0))


def batch_availability(cohort: Cohort, indices: IndexSet) -> np.ndarray:
    return np.asarray(
        [
            [1.0 if cohort.records[int(index)].has(name) else 0.0 for name in MODALITY_STREAMS]
            for index in indices
        ],
        dtype=np.float32,
    )


def move_batch(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Move every tensor of a batch to a device."""
    return TensorBatch(
        streams={name: tensor.to(device) for name, tensor in batch.streams.items()},
        present=batch.present.to(device),
        masks={name: tensor.to(device) for name, tensor in batch.masks.items()},
        record_ids=batch.record_ids,
    )


def describe_batch(batch: TensorBatch) -> Mapping[str, object]:
    return {
        "rows": batch.size(),
        "streams": {name: list(tensor.shape) for name, tensor in batch.streams.items()},
        "visible": list(batch.visible_streams()),
    }
