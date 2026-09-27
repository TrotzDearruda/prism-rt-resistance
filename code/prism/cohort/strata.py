"""Care-setting registry: sites, regions, resource strata and modality budgets.

Ref: Results Sec. 2.6 -- the development partition is made of sites A-E and sites F and G
hold out, with no patient or site overlap; the sites sit in high-resource,
upper-middle-resource and lower-middle-resource settings, and a site in the most
deprived stratum carries fewer modalities. The manuscript names the sites, the four
regions and the three strata but does not print the site-to-region table, so the mapping
below is a reconstruction recorded as such in the claim map.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prism.errors import CohortError
from prism.notation import MODALITY_STREAMS

STRATUM_MODALITIES: Mapping[str, tuple[str, ...]] = {
    "high-resource": ("ct", "mri", "histology", "labs"),
    "upper-middle-resource": ("ct", "mri", "labs"),
    "lower-middle-resource": ("ct", "labs"),
}

STRATUM_CAPACITY: Mapping[str, float] = {
    "high-resource": 1.0,
    "upper-middle-resource": 0.72,
    "lower-middle-resource": 0.45,
}


@dataclass(frozen=True)
class CareSite:
    """One recruiting site with the modalities it can actually acquire."""

    name: str
    region: str
    stratum: str
    availability: tuple[str, ...]

    def superscript(self) -> int:
        return len(self.availability)

    def carries(self, modality: str) -> bool:
        return modality in self.availability


def _site(name: str, region: str, stratum: str) -> CareSite:
    try:
        availability = STRATUM_MODALITIES[stratum]
    except KeyError as failure:
        raise CohortError(f"unknown resource stratum {stratum!r}") from failure
    return CareSite(name=name, region=region, stratum=stratum, availability=availability)


DEFAULT_REGISTRY: tuple[CareSite, ...] = (
    _site("A", "Region I", "high-resource"),
    _site("B", "Region I", "high-resource"),
    _site("C", "Region II", "upper-middle-resource"),
    _site("D", "Region III", "upper-middle-resource"),
    _site("E", "Region IV", "lower-middle-resource"),
    _site("F", "Region III", "lower-middle-resource"),
    _site("G", "Region IV", "lower-middle-resource"),
)


@dataclass(frozen=True)
class Registry:
    """Lookup table over the care settings a study draws on."""

    sites: tuple[CareSite, ...]

    def __post_init__(self) -> None:
        names = [site.name for site in self.sites]
        if len(set(names)) != len(names):
            raise CohortError("site names must be unique")

    def names(self) -> tuple[str, ...]:
        return tuple(site.name for site in self.sites)

    def get(self, name: str) -> CareSite:
        for site in self.sites:
            if site.name == name:
                return site
        raise CohortError(f"no site named {name!r} is registered")

    def regions(self) -> tuple[str, ...]:
        return tuple(sorted({site.region for site in self.sites}))

    def strata(self) -> tuple[str, ...]:
        return tuple(sorted({site.stratum for site in self.sites}))

    def in_region(self, region: str) -> tuple[CareSite, ...]:
        return tuple(site for site in self.sites if site.region == region)

    def in_stratum(self, stratum: str) -> tuple[CareSite, ...]:
        return tuple(site for site in self.sites if site.stratum == stratum)

    def development(self, holdout: tuple[str, ...]) -> tuple[CareSite, ...]:
        return tuple(site for site in self.sites if site.name not in holdout)

    def modality_union(self) -> tuple[str, ...]:
        seen: list[str] = []
        for site in self.sites:
            for modality in site.availability:
                if modality not in seen:
                    seen.append(modality)
        return tuple(seen)

    def capacity(self, name: str) -> float:
        return STRATUM_CAPACITY[self.get(name).stratum]

    def stratum_histogram(self, names: tuple[str, ...]) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for name in names:
            stratum = self.get(name).stratum
            counts[stratum] = counts.get(stratum, 0) + 1
        return counts


def default_registry() -> Registry:
    return Registry(DEFAULT_REGISTRY)


def validate_availability(availability: tuple[str, ...]) -> None:
    unknown = [name for name in availability if name not in MODALITY_STREAMS]
    if unknown:
        raise CohortError(f"unrecognised modality names: {unknown}")


def weakest_stratum(registry: Registry) -> str:
    """The stratum with the smallest modality budget.

    Ref: Results Sec. 2.6 -- the low-resource layer is where the constraint is claimed to
    pay for itself, so it is defined by modality budget rather than by patient count.
    """
    if not registry.sites:
        raise CohortError("an empty registry has no weakest stratum")
    return min(
        registry.strata(),
        key=lambda stratum: (len(STRATUM_MODALITIES[stratum]), stratum),
    )
