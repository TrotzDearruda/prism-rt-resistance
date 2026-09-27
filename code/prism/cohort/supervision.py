"""Metabolite supervision coverage and the batches the alignment term consumes.

Ref: Methods Sec. 6.4 -- the alignment term pulls the fused representation of a region
towards the metabolite block of the same tissue area and matches their second moments, so
a batch is a set of regions rather than a set of patients. Ref: Table 7 lower block -- the
fraction of records carrying spatially resolved supervision is the axis that has not
saturated, which makes the coverage of each region a quantity the study reports rather
than an implementation detail.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.cohort.partitions import stratified_supervision_frame
from prism.cohort.schema import Cohort
from prism.errors import SupervisionError


@dataclass(frozen=True)
class CoverageReport:
    """How much spatially resolved supervision each stratum of the cohort carries."""

    total_records: int
    supervised_records: int
    by_region: Mapping[str, int]
    by_site: Mapping[str, int]
    by_stratum: Mapping[str, int]

    def fraction(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.supervised_records / self.total_records

    def regions_without_supervision(self) -> tuple[str, ...]:
        return tuple(region for region, count in self.by_region.items() if count == 0)

    def thinnest_region(self) -> str:
        if not self.by_region:
            raise SupervisionError("no supervised region is available")
        return min(self.by_region.items(), key=lambda item: (item[1], item[0]))[0]


def coverage_report(cohort: Cohort) -> CoverageReport:
    by_region: dict[str, int] = dict.fromkeys(cohort.regions(), 0)
    by_site: dict[str, int] = dict.fromkeys(cohort.sites(), 0)
    by_stratum: dict[str, int] = dict.fromkeys(cohort.strata(), 0)
    supervised = 0
    for record in cohort:
        if not record.supervised():
            continue
        supervised += 1
        by_region[record.region] = by_region.get(record.region, 0) + 1
        by_site[record.site] = by_site.get(record.site, 0) + 1
        by_stratum[record.stratum] = by_stratum.get(record.stratum, 0) + 1
    return CoverageReport(
        total_records=len(cohort),
        supervised_records=supervised,
        by_region=by_region,
        by_site=by_site,
        by_stratum=by_stratum,
    )


@dataclass(frozen=True)
class AlignmentBatch:
    """Rows of a representation paired with the metabolite block of the same area."""

    rows: np.ndarray
    metabolite: np.ndarray
    regions: tuple[str, ...]
    record_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.rows.shape[0])


def supervised_index(cohort: Cohort) -> tuple[np.ndarray, tuple[str, ...]]:
    """Row positions of supervised records next to their region labels."""
    positions: list[int] = []
    regions: list[str] = []
    for position, record in enumerate(cohort.records):
        if record.supervised():
            positions.append(position)
            regions.append(record.region)
    return np.asarray(positions, dtype=np.int64), tuple(regions)


def alignment_batches(
    cohort: Cohort,
    batch_size: int,
    seed: int,
    *,
    region_balanced: bool = True,
    max_batches: int = 8,
) -> Iterator[AlignmentBatch]:
    """Yield alignment batches drawn from the supervised sub-cohort.

    The default draw is region-balanced, because a region that contributes more
    supervision than another would otherwise dominate the second-moment match.
    """
    if batch_size < 1:
        raise SupervisionError("an alignment batch needs a positive size")
    positions, regions = supervised_index(cohort)
    if positions.size == 0:
        raise SupervisionError("the cohort carries no spatially resolved supervision")
    metabolite = cohort.metabolite_matrix()
    row_lookup = {int(position): row for row, position in enumerate(positions)}
    ordered_regions = tuple(sorted(set(regions)))
    region_members: dict[str, np.ndarray] = {
        region: positions[np.asarray([label == region for label in regions])]
        for region in ordered_regions
    }
    rng = np.random.default_rng(seed)
    if region_balanced:
        per_region = max(1, batch_size // len(ordered_regions))
        take_per_region = batch_size - per_region * (len(ordered_regions) - 1)
    else:
        per_region = 0
        take_per_region = batch_size

    for _ in range(max_batches):
        chosen: list[int] = []
        if region_balanced:
            for position_in_order, region in enumerate(ordered_regions):
                members = region_members[region]
                take = take_per_region if position_in_order == 0 else per_region
                take = min(take, members.size)
                if take == 0:
                    continue
                pick = rng.integers(0, members.size, size=take)
                chosen.extend(int(members[int(index)]) for index in pick)
        else:
            pick = rng.choice(positions, size=min(batch_size, positions.size), replace=False)
            chosen.extend(int(index) for index in pick)
        if not chosen:
            return
        index = np.asarray(sorted(chosen), dtype=np.int64)
        rows = np.asarray([row_lookup[int(position)] for position in index], dtype=np.int64)
        yield AlignmentBatch(
            rows=index,
            metabolite=metabolite[rows],
            regions=tuple(cohort.records[int(position)].region for position in index),
            record_ids=tuple(cohort.records[int(position)].record_id for position in index),
        )


def region_covariance(cohort: Cohort, region: str, *, standardise: bool = True) -> np.ndarray:
    """Metabolite covariance of one region, the target of the second-moment match."""
    rows = [
        np.asarray(record.supervision(), dtype=np.float64)
        for record in cohort
        if record.supervised() and record.region == region
    ]
    if len(rows) < 3:
        raise SupervisionError(
            f"region {region!r} carries {len(rows)} supervised records, too few for a covariance"
        )
    block = np.stack(rows, axis=0)
    if standardise:
        scale = block.std(axis=0, keepdims=True)
        block = block / np.clip(scale, 1e-8, None)
    centred_block: np.ndarray = block - block.mean(axis=0, keepdims=True)
    product: np.ndarray = centred_block.T @ centred_block
    cov: np.ndarray = product / float(block.shape[0] - 1)
    return cov


def supervision_frames(
    cohort: Cohort, fractions: Sequence[float], seed: int
) -> Mapping[float, Cohort]:
    """Supervision fractions for the lower block of Table 7, keyed by the fraction."""
    frames: dict[float, Cohort] = {}
    for fraction in fractions:
        frames[fraction] = stratified_supervision_frame(cohort, fraction, seed)
    return frames


def held_out_region(cohort: Cohort) -> str:
    """The supervised region that is never fitted, used for the coherence read-out."""
    report = coverage_report(cohort)
    return report.thinnest_region()
