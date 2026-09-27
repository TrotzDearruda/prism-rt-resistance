"""Programme coherence.

Ref: Table 7 note -- programme coherence is the agreement between the emitted programme and
the metabolite covariance structure of the held-out supervised region. The quantity exists
so that the lower block of the scaling table can distinguish an emission that has found
metabolic structure from one that has merely become sparse against the same supervision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError


def block_mass_from_covariance(
    covariance: np.ndarray, blocks: Sequence[Sequence[int]]
) -> np.ndarray:
    """Block-wise concentration of a metabolite covariance structure."""
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ConfigError("a metabolite covariance must be square")
    covered = sorted({index for block in blocks for index in block})
    if covered != list(range(values.shape[0])):
        raise ConfigError(
            "the blocks must partition the covariance width, so that a covariance of another "
            "vocabulary cannot be read against this block layout"
        )
    mass = np.zeros(len(blocks), dtype=np.float64)
    for position, block in enumerate(blocks):
        members = np.asarray(list(block), dtype=np.int64)
        submatrix = values[np.ix_(members, members)]
        mass[position] = float(np.sqrt(np.sum(submatrix * submatrix)))
    total = float(mass.sum())
    if total <= 0.0:
        raise ConfigError("the covariance structure carries no mass in any block")
    return mass / total


def block_mass_from_coefficients(
    coefficients: np.ndarray, blocks: Sequence[Sequence[int]]
) -> np.ndarray:
    """Block-wise concentration of an emitted coefficient vector."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ConfigError("coefficients are read as one row per observation")
    width = values.shape[1]
    covered = sorted({index for block in blocks for index in block})
    if covered != list(range(width)):
        raise ConfigError("the blocks must partition the coefficient width")
    mass = np.zeros(len(blocks), dtype=np.float64)
    for position, block in enumerate(blocks):
        members = np.asarray(list(block), dtype=np.int64)
        segment = values[:, members]
        mass[position] = float(np.abs(segment).mean())
    total = float(mass.sum())
    if total <= 0.0:
        raise ConfigError("the emitted coefficients carry no mass in any block")
    return mass / total


def programme_coherence(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> float:
    """Agreement between the emitted programme and the held-out covariance structure.

    The two block mass vectors are already non-negative and unit sum, so their cosine
    similarity is bounded by the overlap of their supports: an emission that concentrates on a
    block the held-out region does not structure scores near zero, and one that concentrates
    on the same blocks as the region scores near one.
    """
    emitted = block_mass_from_coefficients(coefficients, blocks)
    reference = block_mass_from_covariance(covariance, blocks)
    if emitted.size != reference.size:
        raise ConfigError("the emission and the covariance structure must share a block layout")
    denominator = float(np.linalg.norm(emitted) * np.linalg.norm(reference))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(emitted, reference) / denominator)


def retention_of_programmes(
    coefficients: np.ndarray,
    blocks: Sequence[Sequence[int]],
    programme_ids: Sequence[str],
    subset: Sequence[str],
    threshold: float,
) -> float:
    """Share of a named subset of programmes still carried by the emission.

    Ref: Table 7, lower block -- retention of P1-P4 is reported against the fraction of
    records carrying supervision, so retention is read on the named subset rather than over
    every block.
    """
    if len(blocks) != len(programme_ids):
        raise ConfigError("every programme needs one block")
    emitted = block_mass_from_coefficients(coefficients, blocks)
    positions = [programme_ids.index(name) for name in subset if name in programme_ids]
    if not positions:
        raise ConfigError("the named subset contains no programme")
    reference = 1.0 / float(len(positions))
    retained = sum(1 for position in positions if emitted[position] >= threshold * reference)
    return retained / float(len(positions))


@dataclass(frozen=True)
class CoherenceReading:
    """Coherence next to the two mass vectors it was built from."""

    coherence: float
    emitted_mass: tuple[float, ...]
    reference_mass: tuple[float, ...]

    def leading_block(self, programme_ids: Sequence[str]) -> str:
        position = int(np.argmax(np.asarray(self.emitted_mass)))
        return programme_ids[position]

    def as_dict(self) -> Mapping[str, float]:
        values: dict[str, float] = {"coherence": self.coherence}
        for position, mass in enumerate(self.emitted_mass):
            values[f"emitted_mass_block_{position}"] = mass
        for position, mass in enumerate(self.reference_mass):
            values[f"reference_mass_block_{position}"] = mass
        return values


def coherence_reading(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> CoherenceReading:
    emitted = block_mass_from_coefficients(coefficients, blocks)
    reference = block_mass_from_covariance(covariance, blocks)
    return CoherenceReading(
        coherence=programme_coherence(coefficients, covariance, blocks),
        emitted_mass=tuple(float(value) for value in emitted),
        reference_mass=tuple(float(value) for value in reference),
    )


def support_consistency(
    coefficients: np.ndarray, blocks: Sequence[Sequence[int]], threshold: float
) -> float:
    """Share of blocks the emission actually touches.

    A sparse decoder that has collapsed onto one block keeps a high coherence on that block
    alone, so the touched-block share is reported beside the coherence rather than folded
    into it.
    """
    emitted = block_mass_from_coefficients(coefficients, blocks)
    touched = float(np.sum(emitted >= threshold))
    return touched / float(len(blocks))
