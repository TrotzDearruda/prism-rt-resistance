"""Named views over a partitioned cohort.

Ref: Table 1 -- the same model is read on the internal partition, on the external partition and on
the stratum that carries the smallest modality budget, and the three columns are separate read-outs
rather than subsets of one another. This module names those views once so that every table refers
to the same three.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from prism.cohort.schema import Cohort, CohortRecord, Layer
from prism.cohort.strata import STRATUM_MODALITIES, default_registry
from prism.errors import CohortError

INTERNAL = "internal"
EXTERNAL = "external"
LOW_RESOURCE = "low_resource"


@dataclass(frozen=True)
class PartitionView:
    """One named read-out: the records it covers and the labels that go with them."""

    name: str
    cohort: Cohort
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.cohort)

    def rows(self) -> np.ndarray:
        return np.arange(len(self.cohort), dtype=np.int64)

    def prevalence(self) -> float:
        if self.labels.size == 0:
            return 0.0
        return float(self.labels.mean())


def _view(name: str, cohort: Cohort) -> PartitionView:
    return PartitionView(name=name, cohort=cohort, labels=cohort.labels())


def partition_views(parts: Mapping[Layer, Cohort]) -> Mapping[str, PartitionView]:
    """The three views Table 1 reports, each requiring a non-empty record set."""
    if Layer.INTERNAL not in parts or len(parts[Layer.INTERNAL]) == 0:
        raise CohortError("the internal partition is required for the reported comparison")
    if Layer.EXTERNAL not in parts or len(parts[Layer.EXTERNAL]) == 0:
        raise CohortError("the external partition is required for the reported comparison")
    low_stratum = weakest_stratum_from(parts)
    low_cohort = low_resource_cohort(parts, low_stratum)
    return {
        INTERNAL: _view(INTERNAL, parts[Layer.INTERNAL]),
        EXTERNAL: _view(EXTERNAL, parts[Layer.EXTERNAL]),
        LOW_RESOURCE: _view(LOW_RESOURCE, low_cohort),
    }


def weakest_stratum_from(parts: Mapping[Layer, Cohort]) -> str:
    """The stratum with the smallest modality budget across the parts that carry records."""
    combined = combined_cohort(parts)
    if len(combined) == 0:
        raise CohortError("no records are available to find the weakest stratum")
    registry = default_registry()
    known = set(registry.strata())
    observed = sorted(name for name in combined.strata() if name in known)
    if not observed:
        raise CohortError("no recognised resource stratum is represented")
    return min(
        observed,
        key=lambda name: (len(STRATUM_MODALITIES[name]), name),
    )


def low_resource_cohort(parts: Mapping[Layer, Cohort], stratum: str) -> Cohort:
    """Records from the weakest stratum inside the evaluation partitions."""
    evaluation = combined_cohort(
        {
            layer: cohort
            for layer, cohort in parts.items()
            if layer in {Layer.INTERNAL, Layer.EXTERNAL, Layer.SILENT}
        }
    )
    subset = evaluation.by_stratum(stratum)
    if len(subset) == 0:
        raise CohortError(f"the stratum {stratum!r} carries no evaluation records")
    return subset


def combined_cohort(parts: Mapping[Layer, Cohort]) -> Cohort:
    """Every record of the parts, in layer order."""
    collected: list[CohortRecord] = []
    for layer in Layer:
        cohort = parts.get(layer)
        if cohort is None:
            continue
        collected.extend(cohort.records)
    return Cohort(tuple(collected))


def all_evaluation_records(parts: Mapping[Layer, Cohort]) -> Cohort:
    """Internal, external and silent records as one cohort, for a pooled read-out."""
    return combined_cohort(
        {
            layer: cohort
            for layer, cohort in parts.items()
            if layer in {Layer.INTERNAL, Layer.EXTERNAL, Layer.SILENT}
        }
    )


def recruitment_by_layer(parts: Mapping[Layer, Cohort]) -> Mapping[str, int]:
    return {layer.value: len(cohort) for layer, cohort in parts.items()}
