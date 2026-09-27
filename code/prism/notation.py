"""Symbols of the constrained adaptation problem.

Ref: Methods Sec. 6.1. X is the multimodal observation, Y the response, S the care
setting, M the spatially resolved metabolite vector on the supervised sub-cohort,
Z = f(X) the learned representation, g the outcome head and h the programme decoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

MODALITY_STREAMS: tuple[str, ...] = ("ct", "mri", "histology", "labs")

PROGRAMME_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")

PROGRAMME_NAMES: Mapping[str, str] = {
    "P1": "glycolytic",
    "P2": "glutaminolytic",
    "P3": "lipid-desaturation",
    "P4": "hypoxia-redox",
    "P5": "nucleotide",
    "P6": "glutathione-redox-buffering",
}

RESIDUAL_RESOURCE_STRATA: tuple[str, ...] = (
    "high-resource",
    "upper-middle-resource",
    "lower-middle-resource",
)

REGIONS: tuple[str, ...] = ("Region I", "Region II", "Region III", "Region IV")

DEVELOPMENT_SITES: tuple[str, ...] = ("A", "B", "C", "D", "E")
HOLDOUT_SITES: tuple[str, ...] = ("F", "G")


@dataclass(frozen=True)
class Observation:
    """One patient encounter, addressed by the streams that a site could actually supply.

    ``available`` is the site's modality availability list; it is the only place where
    deployment representativeness is expressed, and it is read by the modality-dropout
    sampler rather than stored per record.
    """

    record_id: str
    patient_id: str
    site: str
    region: str
    stratum: str
    outcome: int
    streams: Mapping[str, object] = field(default_factory=dict)

    @property
    def labelled(self) -> bool:
        return self.outcome in (0, 1)


@dataclass(frozen=True)
class MetabolicSupervision:
    """Spatially resolved metabolite abundance for the tissue area of one record.

    Ref: Methods Sec. 6.1, M in R^q where q is the metabolite vocabulary size.
    """

    record_id: str
    abundance: object
    region: str

    def vocabulary_size(self) -> int:
        shape: Sequence[int] = getattr(self.abundance, "shape", ())
        return int(shape[-1]) if shape else 0


@dataclass(frozen=True)
class ProgrammeEmission:
    """The model output: a supported metabolite set with signs and effect estimates.

    Ref: Methods Sec. 6.1 and Sec. 6.6. The output is not a risk score; it is a
    programme ``pi`` together with a per-programme effect ``beta`` and interval ``I``.
    """

    record_id: str
    supported: tuple[str, ...]
    signs: Mapping[str, int]
    programme_support: tuple[str, ...]
    beta: Mapping[str, float]
    interval: Mapping[str, tuple[float, float]]

    def n_supported(self) -> int:
        return len(self.supported)
