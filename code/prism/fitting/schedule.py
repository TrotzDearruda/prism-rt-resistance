"""Learning-rate schedule.

Ref: Algorithm 1 -- the schedule is declared before any model is fitted. The warmup fraction is
expressed as a share of the total step budget rather than as an absolute count so that the same
configuration file describes both stages without being rewritten, and the decay is a cosine
that ends at a small floor rather than at zero, because a zero rate in the last steps of the
constrained stage would freeze the feasibility check rather than settle it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from prism.errors import ConfigError


@dataclass(frozen=True)
class ScheduleSettings:
    """Warmup share, final floor and total step budget."""

    total_steps: int
    warmup_fraction: float
    final_factor: float = 0.01

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ConfigError("a schedule needs at least one step")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ConfigError("a warmup fraction must lie in [0, 1)")
        if not 0.0 < self.final_factor <= 1.0:
            raise ConfigError("a final factor must lie in (0, 1]")

    def warmup_steps(self) -> int:
        self.validate()
        return int(round(self.warmup_fraction * self.total_steps))


def warmup_cosine_factor(step: int, settings: ScheduleSettings) -> float:
    """Multiplier applied to the group rates at one step."""
    settings.validate()
    warmup = settings.warmup_steps()
    if step < warmup:
        return float(step + 1) / float(max(warmup, 1))
    span = max(settings.total_steps - warmup, 1)
    progress = min(1.0, float(step - warmup) / float(span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return settings.final_factor + (1.0 - settings.final_factor) * cosine


def schedule_factors(settings: ScheduleSettings) -> tuple[float, ...]:
    return tuple(warmup_cosine_factor(step, settings) for step in range(settings.total_steps))


def make_lambda(settings: ScheduleSettings) -> Callable[[int], float]:
    """A callable for a torch learning-rate scheduler."""
    settings.validate()

    def factor(step: int) -> float:
        return warmup_cosine_factor(step, settings)

    return factor


def attach_scheduler(
    optimiser: torch.optim.Optimizer, settings: ScheduleSettings
) -> torch.optim.lr_scheduler.LambdaLR:
    """Wrap an optimiser in the schedule, leaving the base rates untouched."""
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=make_lambda(settings))


def current_rates(optimiser: torch.optim.Optimizer) -> Mapping[str, float]:
    return {
        str(group.get("name", f"group{position}")): float(group["lr"])
        for position, group in enumerate(optimiser.param_groups)
    }


def steps_per_epoch(n_rows: int, batch_size: int, grad_accum: int) -> int:
    """Optimiser steps in one epoch, counting an accumulation window as one step."""
    if batch_size < 1 or grad_accum < 1:
        raise ConfigError("batch size and accumulation window must be positive")
    if n_rows < 0:
        raise ConfigError("a row count cannot be negative")
    windows = math.ceil(n_rows / float(batch_size))
    return max(1, math.ceil(windows / float(grad_accum)))


def total_steps_for_stages(
    n_rows: int, batch_size: int, grad_accum: int, stage1_epochs: int, stage2_epochs: int
) -> tuple[int, int]:
    per_epoch = steps_per_epoch(n_rows, batch_size, grad_accum)
    return per_epoch * stage1_epochs, per_epoch * stage2_epochs


def cosine_with_restarts(base: float, restart_every: int, n_steps: int) -> Sequence[float]:
    """Rates for a run that is restarted, reported when a constrained stage is extended."""
    if restart_every < 1 or n_steps < 1:
        raise ConfigError("a restart schedule needs a positive window and budget")
    rates: list[float] = []
    for step in range(n_steps):
        position = step % restart_every
        progress = float(position) / float(restart_every)
        rates.append(base * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return tuple(rates)
