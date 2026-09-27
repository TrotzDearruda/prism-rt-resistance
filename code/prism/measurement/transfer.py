"""Cross-region and cross-domain transfer.

Ref: Table 3 Panel A -- a model is fitted on the region named by the row and evaluated on the
region named by the column, with a prediction interval carried in every cell so that a
cell's width can be read against its effect without a lookup. The panel also reports a
pooled site-invariant row whose entries may exceed some diagonal cells, and the manuscript
states that this is reported as observed rather than smoothed.
Ref: Table 6 -- the same machinery is applied without task-specific re-parameterisation to
adjacent tumour types and to one non-colorectal setting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError


@dataclass(frozen=True)
class TransferCell:
    """One fitted-on/evaluated-on pair."""

    fitted_on: str
    evaluated_on: str
    auc: float
    interval: tuple[float, float]

    def half_width(self) -> float:
        return 0.5 * (self.interval[1] - self.interval[0])

    def as_dict(self) -> Mapping[str, float | str | tuple[float, float]]:
        return {
            "fitted_on": self.fitted_on,
            "evaluated_on": self.evaluated_on,
            "auc": self.auc,
            "interval_lower": self.interval[0],
            "interval_upper": self.interval[1],
        }


@dataclass(frozen=True)
class TransferMatrix:
    """A square transfer matrix with a pooled site-invariant row."""

    rows: tuple[str, ...]
    pooled_key: str
    cells: tuple[TransferCell, ...]

    def auc(self, fitted_on: str, evaluated_on: str) -> float:
        for cell in self.cells:
            if cell.fitted_on == fitted_on and cell.evaluated_on == evaluated_on:
                return cell.auc
        raise ConfigError(f"no cell for {fitted_on!r} -> {evaluated_on!r}")

    def row(self, fitted_on: str) -> tuple[TransferCell, ...]:
        return tuple(cell for cell in self.cells if cell.fitted_on == fitted_on)

    def diagonal(self) -> tuple[TransferCell, ...]:
        return tuple(cell for cell in self.cells if cell.fitted_on == cell.evaluated_on)

    def retention(self, fitted_on: str) -> float:
        """Smallest off-diagonal over diagonal ratio for one fitted row.

        Ref: Results Sec. 2.2 -- the retention factor is the quantity read off the matrix
        when the diagonal is compared with the off-diagonal entries of the same row.
        """
        diagonal = self.auc(fitted_on, fitted_on)
        off = [cell.auc for cell in self.row(fitted_on) if cell.evaluated_on != fitted_on]
        if not off or diagonal <= 0.0:
            raise ConfigError("a retention factor needs a diagonal and an off-diagonal entry")
        return min(off) / diagonal

    def smallest_retention(self) -> tuple[str, float]:
        factors = [(fitted, self.retention(fitted)) for fitted in self.rows]
        return min(factors, key=lambda item: item[1])

    def pooled_exceeds_diagonal(self) -> tuple[tuple[str, float, float], ...]:
        """Cells where the pooled row beats the region's own diagonal.

        Ref: Table 3 caption -- the diagonal is not uniformly the maximum, and the deviation
        is reported as observed.
        """
        if not any(cell.fitted_on == self.pooled_key for cell in self.cells):
            return ()
        findings: list[tuple[str, float, float]] = []
        for fitted in self.rows:
            diagonal = self.auc(fitted, fitted)
            pooled = self.auc(self.pooled_key, fitted)
            if pooled > diagonal:
                findings.append((fitted, pooled, diagonal))
        return tuple(findings)

    def as_table(self) -> Mapping[str, Mapping[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for cell in self.cells:
            table.setdefault(cell.fitted_on, {})[cell.evaluated_on] = cell.auc
        return table


def build_matrix(
    rows: Sequence[str],
    auc_by_pair: Mapping[tuple[str, str], float],
    interval_half_width: float,
    pooled_key: str | None = None,
) -> TransferMatrix:
    """Assemble the matrix, building each interval from a stated half-width."""
    if not rows:
        raise ConfigError("a transfer matrix needs at least one row")
    cells: list[TransferCell] = []
    for fitted in rows:
        for evaluated in rows:
            key = (fitted, evaluated)
            if key not in auc_by_pair:
                raise ConfigError(f"the matrix is missing the pair {key}")
            auc = float(auc_by_pair[key])
            cells.append(
                TransferCell(
                    fitted_on=fitted,
                    evaluated_on=evaluated,
                    auc=auc,
                    interval=(auc - interval_half_width, auc + interval_half_width),
                )
            )
    if pooled_key is not None:
        for evaluated in rows:
            key = (pooled_key, evaluated)
            if key not in auc_by_pair:
                raise ConfigError(f"the pooled row is missing the column {evaluated!r}")
            auc = float(auc_by_pair[key])
            cells.append(
                TransferCell(
                    fitted_on=pooled_key,
                    evaluated_on=evaluated,
                    auc=auc,
                    interval=(auc - interval_half_width, auc + interval_half_width),
                )
            )
    return TransferMatrix(rows=tuple(rows), pooled_key=pooled_key or "", cells=tuple(cells))


@dataclass(frozen=True)
class TransferAdvantage:
    """One row of the cross-domain comparison of Table 6."""

    setting: str
    unconstrained_auc: float
    constrained_auc: float
    linear_probe_auc: float
    metabolic_supervision_auc: float

    def advantage(self) -> float:
        return self.constrained_auc - self.unconstrained_auc

    def probe_gap(self) -> float:
        return self.constrained_auc - self.linear_probe_auc

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "setting": self.setting,
            "unconstrained_auc": self.unconstrained_auc,
            "constrained_auc": self.constrained_auc,
            "advantage": self.advantage(),
            "linear_probe_auc": self.linear_probe_auc,
            "metabolic_supervision_auc": self.metabolic_supervision_auc,
        }


def advantage_reading(rows: Sequence[TransferAdvantage]) -> Mapping[str, float]:
    """Where the advantage is largest and how it decays away from the primary task.

    Ref: Results Sec. 3.1 -- the advantage of the constrained configuration is largest on the
    primary task and falls off for the adjacent settings, which the manuscript reads as the
    invariance constraint paying for itself where acquisition variability is largest.
    """
    if not rows:
        raise ConfigError("an advantage reading needs at least one setting")
    primary = rows[0]
    others = rows[1:]
    decay = [primary.advantage() - row.advantage() for row in others]
    return {
        "primary_advantage": primary.advantage(),
        "mean_adjacent_advantage": (
            float(np.mean([row.advantage() for row in others])) if others else 0.0
        ),
        "mean_decay": float(np.mean(decay)) if decay else 0.0,
        "probe_gap_primary": primary.probe_gap(),
    }


def probe_only_row(rows: Sequence[TransferAdvantage]) -> Mapping[str, float]:
    """The probe column read on its own.

    Ref: Table 6 note -- the probe column isolates the transferable content of the frozen
    representation from the constrained fine-tuning, so the contribution of the objective is
    not confused with the contribution of pretraining.
    """
    if not rows:
        raise ConfigError("a probe reading needs at least one setting")
    gains = [row.constrained_auc - row.linear_probe_auc for row in rows]
    return {
        "mean_probe_gap": float(np.mean(gains)),
        "min_probe_gap": float(np.min(gains)),
        "max_probe_gap": float(np.max(gains)),
    }
