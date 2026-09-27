"""Seed control.

Ref: Algorithm 1 notes that all hyper-parameters are declared before any model is
fitted; the fit ensemble is repeated over random seeds, so the seed is part of the
run identity rather than an incidental detail.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_SEED = 20240921
SEED_ENSEMBLE: tuple[int, ...] = (20240921, 20240922, 20240923, 20240924, 20240925)


@dataclass(frozen=True)
class SeedState:
    """The seed a run was launched with, carried into every artefact it writes."""

    seed: int
    deterministic: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {"seed": self.seed, "deterministic": self.deterministic}


def set_seed(seed: int, *, deterministic: bool = False) -> SeedState:
    """Seed python, numpy and torch, and return the state to record in artefacts."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return SeedState(seed=seed, deterministic=deterministic)


def ensemble_seeds(n_members: int, base: int = DEFAULT_SEED) -> tuple[int, ...]:
    """Seeds for an n-member ensemble, derived from the run seed without repetition."""
    if n_members < 1:
        raise ValueError("an ensemble needs at least one member")
    return tuple(base + 101 * i for i in range(n_members))
