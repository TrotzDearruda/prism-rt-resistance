"""The null band.

Ref: Methods Sec. 6.10 -- the band with no result, used to read the ablation table, is
T = 0.003 expressed as a fraction. The caption of the ablation table adds two rules on top of
it: no ablation in the table moves a metric by less than the band except the interval-head row,
which is reported as null on discrimination and as a coverage failure on calibration. Both
rules are expressed here so that the table cannot be read by eye.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prism.errors import ConfigError

NULL_BAND = 0.003


@dataclass(frozen=True)
class NullBand:
    """A threshold below which a measured movement is reported as no movement."""

    value: float = NULL_BAND

    def validate(self) -> None:
        if self.value <= 0.0:
            raise ConfigError("a null band must be positive")

    def resolves(self, movement: float) -> bool:
        """Whether a movement is larger than the band, in either direction."""
        self.validate()
        return abs(float(movement)) > self.value

    def direction(self, movement: float) -> str:
        self.validate()
        if abs(float(movement)) <= self.value:
            return "null"
        return "increase" if movement > 0.0 else "decrease"


@dataclass(frozen=True)
class AblationMovement:
    """One ablation row read against the full configuration."""

    configuration: str
    column: str
    movement: float
    resolves: bool

    def as_dict(self) -> Mapping[str, float | str | bool]:
        return {
            "configuration": self.configuration,
            "column": self.column,
            "movement": self.movement,
            "resolves_against_null_band": self.resolves,
        }


def read_ablation(
    configuration: str,
    column: str,
    full_value: float,
    ablated_value: float,
    band: NullBand | None = None,
) -> AblationMovement:
    """Movement of one cell away from the full configuration."""
    resolved = band if band is not None else NullBand()
    movement = float(ablated_value) - float(full_value)
    return AblationMovement(
        configuration=configuration,
        column=column,
        movement=movement,
        resolves=resolved.resolves(movement),
    )


def null_on_discrimination(movement: float, band: NullBand | None = None) -> bool:
    """Whether a movement is reportable as no movement on a discrimination column."""
    resolved = band if band is not None else NullBand()
    return not resolved.resolves(movement)


def coverage_failure(nominal: float, empirical: float, band: NullBand | None = None) -> bool:
    """Whether an interval head failed on calibration rather than on discrimination.

    Ref: Table 2 caption -- the interval-head row is reported as null on discrimination and as
    a coverage failure on calibration, so the two readings are carried separately instead of
    being collapsed into one verdict for the row.
    """
    resolved = band if band is not None else NullBand()
    gap = float(nominal) - float(empirical)
    return resolved.resolves(gap) and gap > 0.0


@dataclass(frozen=True)
class NullBandLedger:
    """Every movement a table reported, next to the band that was used to read it."""

    band: NullBand
    movements: tuple[AblationMovement, ...]

    def unresolved(self) -> tuple[AblationMovement, ...]:
        return tuple(movement for movement in self.movements if not movement.resolves)

    def resolved(self) -> tuple[AblationMovement, ...]:
        return tuple(movement for movement in self.movements if movement.resolves)

    def columns_all_null(self) -> tuple[str, ...]:
        by_column: dict[str, list[bool]] = {}
        for movement in self.movements:
            by_column.setdefault(movement.column, []).append(movement.resolves)
        return tuple(column for column, flags in by_column.items() if not any(flags))

    def as_table(self) -> tuple[Mapping[str, float | str | bool], ...]:
        return tuple(movement.as_dict() for movement in self.movements)


def band_from_config(value: float) -> NullBand:
    band = NullBand(value=float(value))
    band.validate()
    return band
