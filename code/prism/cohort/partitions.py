"""Patient-level partitioning.

Ref: Results Sec. 2.6 -- partitioning happens on a patient basis, the development
partition is made of sites A-E while sites F and G hold out, and every cross-setting
number is produced from a site-stratified partition rather than from a random patient
split. Ref: Results Sec. 2.1 -- 3,847 held-out records sit apart from the development arm
and a further 1,024 records form the silent exposure sample.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from prism.cohort.schema import Cohort, CohortRecord, Layer
from prism.errors import CohortError


@dataclass(frozen=True)
class PartitionSpec:
    """Target layer sizes and the site boundary between development and holdout.

    Ref: Results Sec. 2.6 -- the unit of every count is the patient, because a patient is the unit
    of partitioning and of resampling. A caller holding record counts converts before building a
    spec, so a cohort where a patient contributes several encounters cannot silently produce a
    development block that is twice its intended size.
    """

    development_patients: int
    internal_patients: int
    external_patients: int
    silent_patients: int
    development_sites: tuple[str, ...]
    holdout_sites: tuple[str, ...]
    external_regions: tuple[str, ...]

    def total(self) -> int:
        return (
            self.development_patients
            + self.internal_patients
            + self.external_patients
            + self.silent_patients
        )


def _patient_order(cohort: Cohort, seed: int) -> list[str]:
    patients = sorted(set(cohort.patient_ids()))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(patients))
    return [patients[int(index)] for index in permutation]


def assign_layers(cohort: Cohort, spec: PartitionSpec, seed: int) -> Mapping[str, Layer]:
    """Map every patient to exactly one layer.

    The development block is filled from the development sites only; the external block
    is filled from the holdout sites only, which is what makes the external estimate
    independent of the development arm by construction rather than by assumption.
    """
    order = _patient_order(cohort, seed)
    record_by_patient: dict[str, CohortRecord] = {}
    for record in cohort:
        record_by_patient.setdefault(record.patient_id, record)

    development_pool = [
        key for key in order if record_by_patient[key].site in spec.development_sites
    ]
    holdout_pool = [key for key in order if record_by_patient[key].site in spec.holdout_sites]
    if not development_pool:
        raise CohortError("no patient sits at a development site")
    if not holdout_pool:
        raise CohortError("no patient sits at a holdout site")

    assignment: dict[str, Layer] = {}
    n_development = min(spec.development_patients, len(development_pool))
    for key in development_pool[:n_development]:
        assignment[key] = Layer.DEVELOPMENT
    for key in development_pool[n_development:]:
        assignment[key] = Layer.INTERNAL
    n_external = min(spec.external_patients, len(holdout_pool))
    for key in holdout_pool[:n_external]:
        assignment[key] = Layer.EXTERNAL
    for key in holdout_pool[n_external:]:
        assignment[key] = Layer.SILENT
    return assignment


def partition_cohort(cohort: Cohort, spec: PartitionSpec, seed: int) -> dict[Layer, Cohort]:
    """Split a cohort into layers, keeping every patient whole."""
    assignment = assign_layers(cohort, spec, seed)
    buckets: dict[Layer, list[CohortRecord]] = {layer: [] for layer in Layer}
    for record in cohort:
        buckets[assignment[record.patient_id]].append(record)
    result = {layer: Cohort(tuple(records)) for layer, records in buckets.items()}
    check_no_patient_overlap(result)
    check_site_boundary(result, spec)
    return result


def check_no_patient_overlap(parts: Mapping[Layer, Cohort]) -> None:
    seen: dict[str, Layer] = {}
    for layer, cohort in parts.items():
        for patient in cohort.patient_ids():
            previous = seen.get(patient)
            if previous is not None and previous is not layer:
                raise CohortError(
                    f"patient {patient} appears in both {previous.value} and {layer.value}"
                )
            seen[patient] = layer


def check_site_boundary(parts: Mapping[Layer, Cohort], spec: PartitionSpec) -> None:
    development_sites = {record.site for record in parts[Layer.DEVELOPMENT]}
    internal_sites = {record.site for record in parts[Layer.INTERNAL]}
    external_sites = {record.site for record in parts[Layer.EXTERNAL]}
    overlapping = (development_sites | internal_sites) & external_sites
    if overlapping:
        raise CohortError(
            "external evaluation shares sites with the development arm: " f"{sorted(overlapping)}"
        )
    if not development_sites <= set(spec.development_sites):
        raise CohortError("a development record sits at a site that was not released")


def fold_by_site(cohort: Cohort) -> Iterator[tuple[str, Cohort, Cohort]]:
    """Leave-one-site-out folds, the validation scheme for the interval head.

    Ref: Algorithm 1 step 16 -- the calibrated interval head is validated
    leave-one-site-out.
    """
    for site in cohort.sites():
        held = cohort.by_site(site)
        rest = Cohort(tuple(record for record in cohort if record.site != site))
        if len(held) == 0 or len(rest) == 0:
            continue
        yield site, rest, held


def fold_by_region(cohort: Cohort) -> Iterator[tuple[str, Cohort, Cohort]]:
    """Leave-one-region-out folds, used for the cross-region transfer matrix.

    Ref: Table 3 Panel A -- a model fitted on one region is evaluated on each other
    region, so the fold definition is geographic rather than random.
    """
    for region in cohort.regions():
        held = cohort.by_region(region)
        rest = Cohort(tuple(record for record in cohort if record.region != region))
        if len(held) == 0 or len(rest) == 0:
            continue
        yield region, rest, held


def training_frame(cohort: Cohort, fraction: float, seed: int) -> Cohort:
    """Sub-sample the fitting arm by patient, for the scaling curves of Table 7."""
    if not 0.0 < fraction <= 1.0:
        raise CohortError("a fitting fraction must lie in (0, 1]")
    if fraction == 1.0:
        return cohort
    order = _patient_order(cohort, seed)
    keep = set(order[: max(1, int(round(fraction * len(order))))])
    return Cohort(tuple(record for record in cohort if record.patient_id in keep))


def stratified_supervision_frame(cohort: Cohort, fraction: float, seed: int) -> Cohort:
    """Keep a fraction of the supervised records, for the lower block of Table 7.

    Ref: Table 7 lower block -- external discrimination and programme coherence against
    the fraction of records carrying spatially resolved metabolite supervision.
    """
    if not 0.0 < fraction <= 1.0:
        raise CohortError("a supervision fraction must lie in (0, 1]")
    supervised = cohort.supervised()
    if fraction == 1.0 or len(supervised) == 0:
        return cohort
    order = _patient_order(supervised, seed)
    keep = set(order[: max(1, int(round(fraction * len(order))))])
    chosen = {record.record_id for record in supervised if record.patient_id in keep}
    rebuilt: list[CohortRecord] = []
    for record in cohort:
        if record.supervised() and record.record_id not in chosen:
            rebuilt.append(
                CohortRecord(
                    record_id=record.record_id,
                    patient_id=record.patient_id,
                    site=record.site,
                    region=record.region,
                    stratum=record.stratum,
                    outcome=record.outcome,
                    layer=record.layer,
                    features=record.features,
                    availability=record.availability,
                    metabolite=None,
                )
            )
        else:
            rebuilt.append(record)
    return Cohort(tuple(rebuilt))
