"""Fixture records for the execution checks.

The release contains no cohort generator: a run reads the arm the data owner staged, and the
clinical arm is not redistributable. The checks still have to execute the code paths, so this module
builds a small set of records with the same field layout -- patients, care settings, resource
strata, streams at the configured shapes, response labels and supervised rows -- and hands them to
the probes and to the test suite. Nothing here stands in for a study cohort, no value produced from
it is reported as a result, and no module of the library imports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism.cohort.lexicon import Lexicon, default_lexicon
from prism.cohort.partitions import PartitionSpec, partition_cohort
from prism.cohort.schema import Cohort, CohortRecord, Layer
from prism.cohort.strata import default_registry
from prism.errors import CohortError
from prism.notation import MODALITY_STREAMS, PROGRAMME_IDS
from prism.provenance.config import ExperimentConfig, stream_shapes

FIXTURE_SEED = 20240921
RECORDS_PER_PATIENT = 2


@dataclass(frozen=True)
class FixtureSpec:
    """How many records a fixture carries and how much of it is supervised."""

    n_records: int = 320
    n_silent: int = 80
    prevalence: float = 0.194
    supervision_fraction: float = 0.5
    records_per_patient: int = RECORDS_PER_PATIENT
    seed: int = FIXTURE_SEED

    def n_patients(self) -> int:
        if self.n_records < 2 or self.records_per_patient < 1:
            raise CohortError("a fixture needs at least one patient with two records")
        return int(np.ceil(self.n_records / self.records_per_patient))


@dataclass(frozen=True)
class Fixture:
    """The records, the vocabulary they are indexed by, and their layers."""

    cohort: Cohort
    lexicon: Lexicon
    parts: Mapping[Layer, Cohort]
    spec: FixtureSpec

    def development(self) -> Cohort:
        return self.parts[Layer.DEVELOPMENT]

    def held_out(self) -> Cohort:
        internal = self.parts.get(Layer.INTERNAL)
        if internal is not None and len(internal) > 0:
            return internal
        return self.parts[Layer.DEVELOPMENT]


def _offered_site(rng: np.random.Generator, weights: tuple[float, ...]) -> int:
    drawn: int = rng.choice(len(weights), p=weights)
    return int(drawn)


def build_fixture(
    shapes: Mapping[str, tuple[int, ...]],
    spec: FixtureSpec | None = None,
    lexicon: Lexicon | None = None,
) -> Fixture:
    """Build a fixture whose streams carry the shapes a configuration declares."""
    resolved = spec if spec is not None else FixtureSpec()
    resolved_lexicon = lexicon if lexicon is not None else default_lexicon()
    missing = [name for name in resolved_lexicon.programme_ids if name not in PROGRAMME_IDS]
    if missing:
        raise CohortError(
            f"the vocabulary names programmes this release does not define: {missing}"
        )

    registry = default_registry()
    sites = registry.names()
    weights = np.asarray([registry.capacity(name) for name in sites], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(resolved.seed)
    n_supervised = int(round(resolved.supervision_fraction * resolved.n_records))

    programme_loadings = rng.normal(
        0.0, 1.0, size=(resolved_lexicon.size, len(resolved_lexicon.programme_ids))
    )
    for position, members in enumerate(resolved_lexicon.index_blocks()):
        programme_loadings[list(members), position] += 1.2
    effects = np.linspace(1.0, 0.3, len(resolved_lexicon.programme_ids))

    records: list[CohortRecord] = []
    for patient in range(resolved.n_patients()):
        site = sites[_offered_site(rng, weights)]
        care = registry.get(site)
        activation = rng.gamma(shape=1.4, scale=0.7, size=len(resolved_lexicon.programme_ids))
        rate = 1.0 / (1.0 + np.exp(-(float(effects @ activation) - 1.6)))
        outcome = int(rng.random() < rate)
        for encounter in range(resolved.records_per_patient):
            position = patient * resolved.records_per_patient + encounter
            if position >= resolved.n_records:
                break
            features: dict[str, np.ndarray] = {}
            for name in care.availability:
                if name not in shapes:
                    raise CohortError(
                        f"the configuration declares no shape for the {name} stream that site "
                        f"{site} supplies"
                    )
                size = int(np.prod(shapes[name]))
                projection = rng.normal(0.0, 1.0, size=(size, activation.size))
                block = (projection @ activation + rng.normal(0.0, 0.3, size=size)).reshape(
                    shapes[name]
                )
                features[name] = block
            metabolite: np.ndarray | None = None
            if position < n_supervised:
                metabolite = programme_loadings @ activation + rng.normal(
                    0.0, 0.4, size=resolved_lexicon.size
                )
            records.append(
                CohortRecord(
                    record_id=f"F{position:06d}",
                    patient_id=f"P{patient:06d}",
                    site=site,
                    region=care.region,
                    stratum=care.stratum,
                    outcome=outcome,
                    layer=Layer.DEVELOPMENT,
                    features=features,
                    availability=care.availability,
                    metabolite=metabolite,
                )
            )

    cohort = Cohort(tuple(records))
    developmental = tuple(name for name in sites if name not in registry.names()[-2:])
    partition = PartitionSpec(
        development_patients=max(1, int(0.70 * resolved.n_patients())),
        internal_patients=max(1, resolved.n_patients() - int(0.70 * resolved.n_patients())),
        external_patients=max(1, resolved.n_patients() // 4),
        silent_patients=0,
        development_sites=developmental,
        holdout_sites=registry.names()[-2:],
        external_regions=registry.regions()[-2:],
    )
    parts = partition_cohort(cohort, partition, resolved.seed)
    # The cohort the fixture reports carries the assigned layer on each record, so staging it as an
    # archive exercises the layer field as well as the streams.
    layered = Cohort(tuple(record for layer in Layer for record in parts[layer].records))
    return Fixture(cohort=layered, lexicon=resolved_lexicon, parts=parts, spec=resolved)


def fixture_for_config(config: ExperimentConfig, spec: FixtureSpec | None = None) -> Fixture:
    """A fixture sized and shaped the way one configuration declares.

    A run checks the arm it reads against the record count the configuration states, so the records
    a probe stages have to carry that count as well as the declared stream shapes.
    """
    declared = config.cohort.n_records + config.cohort.n_silent
    resolved = (
        spec
        if spec is not None
        else FixtureSpec(
            n_records=declared,
            n_silent=config.cohort.n_silent,
            prevalence=config.cohort.prevalence,
            supervision_fraction=config.supervision.fraction,
            seed=config.train.seed,
        )
    )
    return build_fixture(stream_shapes(config.model), resolved)


def available_modalities(cohort: Cohort) -> tuple[str, ...]:
    """The streams a fixture's sites can supply, in the release's order."""
    seen: list[str] = []
    for record in cohort:
        for name in record.availability:
            if name in MODALITY_STREAMS and name not in seen:
                seen.append(name)
    return tuple(seen)


def write_fixture_archive(cohort: Cohort, path: Path, metabolite_width: int) -> Path:
    """Stage a fixture as an archive, so the archive reader itself can be executed."""
    from prism.cohort.archive import write_archive

    write_archive(
        cohort,
        path,
        metabolite_width=metabolite_width,
        provenance="fixture records built by the verification harness; not study data",
    )
    return path
