"""Feasibility constraint.

Ref: Methods Sec. 6.1 Eq. (2) and Algorithm 1 step 9-11 -- the constrained problem carries
D(Y, g o f) >= D_min, and an epoch whose update would breach the floor is reverted. The
constraint is what keeps the invariance penalty from being satisfied by a representation
that has thrown away the response along with the setting; the manuscript makes that point
explicitly, so the feasibility logic is a first-class part of the objective rather than a
training trick.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from prism.errors import ConfigError, FeasibilityError


def state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Detached copy of a module's parameters, used to revert an epoch."""
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def restore_snapshot(module: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    module.load_state_dict(dict(snapshot))


@dataclass(frozen=True)
class FeasibilityFloor:
    """The pre-declared discrimination floor and the slack allowed around it."""

    d_min: float
    slack: float = 1e-4

    def admissible(self, discrimination: float) -> bool:
        return discrimination >= self.d_min - self.slack

    def shortfall(self, discrimination: float) -> float:
        return max(0.0, self.d_min - discrimination)


@dataclass(frozen=True)
class FeasibilityVerdict:
    """What one epoch did to the floor."""

    epoch: int
    discrimination: float
    floor: float
    admitted: bool
    shortfall: float

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "epoch": self.epoch,
            "discrimination": self.discrimination,
            "floor": self.floor,
            "admitted": self.admitted,
            "shortfall": self.shortfall,
        }


@dataclass
class FeasibilityLedger:
    """Record of every veto, so a run can report how often the floor bound it."""

    floor: FeasibilityFloor
    verdicts: list[FeasibilityVerdict] = field(default_factory=list)

    def record(self, epoch: int, discrimination: float) -> FeasibilityVerdict:
        verdict = FeasibilityVerdict(
            epoch=epoch,
            discrimination=discrimination,
            floor=self.floor.d_min,
            admitted=self.floor.admissible(discrimination),
            shortfall=self.floor.shortfall(discrimination),
        )
        self.verdicts.append(verdict)
        return verdict

    def n_vetoes(self) -> int:
        return sum(1 for verdict in self.verdicts if not verdict.admitted)

    def worst(self) -> FeasibilityVerdict | None:
        if not self.verdicts:
            return None
        return min(self.verdicts, key=lambda verdict: verdict.discrimination)

    def summary(self) -> Mapping[str, float]:
        return {
            "epochs": float(len(self.verdicts)),
            "vetoes": float(self.n_vetoes()),
            "floor": self.floor.d_min,
        }


def evaluate_epoch(
    module: nn.Module,
    epoch: int,
    discrimination: float,
    floor: FeasibilityFloor,
    snapshot: Mapping[str, torch.Tensor],
) -> FeasibilityVerdict:
    """Admit an epoch or revert it, returning the verdict either way."""
    if discrimination < 0.0 or discrimination > 1.0:
        raise ConfigError("a discrimination value outside [0, 1] cannot come from the design")
    admitted = floor.admissible(discrimination)
    if not admitted:
        restore_snapshot(module, snapshot)
    return FeasibilityVerdict(
        epoch=epoch,
        discrimination=discrimination,
        floor=floor.d_min,
        admitted=admitted,
        shortfall=floor.shortfall(discrimination),
    )


def assert_feasible_at_end(ledger: FeasibilityLedger) -> None:
    """A run that never cleared the floor is reported rather than silently shipped."""
    if not ledger.verdicts:
        raise FeasibilityError("no epoch was evaluated against the discrimination floor")
    if all(not verdict.admitted for verdict in ledger.verdicts):
        worst = ledger.worst()
        assert worst is not None
        raise FeasibilityError(
            "every epoch breached the discrimination floor; the worst shortfall was "
            f"{worst.shortfall:.4f}"
        )


def trade_off_point(
    admitted: bool, discrimination: float, floor: FeasibilityFloor
) -> Mapping[str, float]:
    """Position of an admitted update on the invariance-discrimination frontier.

    Ref: Methods Sec. 6.1 -- the pair (D_min, lambda_I) defines a point on the frontier, so a
    run reports where it landed rather than only whether it passed.
    """
    if not admitted:
        return {"on_frontier": 0.0, "margin": 0.0}
    return {
        "on_frontier": 1.0,
        "margin": discrimination - floor.d_min,
    }
