"""Shared fixtures for the test suite.

Every fixture builds its subject from a small, deterministic configuration so that a test failure
points at one behaviour rather than at a schedule. The records come from the harness builder in
``checks.fixtures``: the release contains no cohort generator, and the clinical arm is not
redistributable, so the test suite supplies its own records and reports no result from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from checks.fixtures import Fixture, FixtureSpec, fixture_for_config
from prism.cohort.lexicon import Lexicon, default_lexicon
from prism.cohort.schema import Cohort, Layer
from prism.modelling.assembly import Prism
from prism.objectives.total import ConstrainedObjective, ObjectiveWeights
from prism.provenance.config import ExperimentConfig, load_experiment, parse_config
from prism.provenance.paths import RunPaths, run_paths

SMOKE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "_smoke.yaml"


@pytest.fixture(scope="session")
def smoke_config() -> ExperimentConfig:
    return load_experiment(SMOKE_CONFIG)


@pytest.fixture(scope="session")
def lexicon() -> Lexicon:
    return default_lexicon()


@pytest.fixture(scope="session")
def records(smoke_config: ExperimentConfig) -> Fixture:
    return fixture_for_config(
        smoke_config, FixtureSpec(n_records=200, n_silent=40, supervision_fraction=0.5, seed=11)
    )


@pytest.fixture(scope="session")
def parts(records: Fixture) -> Mapping[Layer, Cohort]:
    return dict(records.parts)


@pytest.fixture(scope="session")
def development(parts: Mapping[Layer, Cohort]) -> Cohort:
    return parts[Layer.DEVELOPMENT]


@pytest.fixture(scope="session")
def model(smoke_config: ExperimentConfig, lexicon: Lexicon) -> Prism:
    torch.manual_seed(smoke_config.train.seed)
    return Prism(smoke_config.model, lexicon, arm=smoke_config.arm)


@pytest.fixture()
def objective() -> ConstrainedObjective:
    return ConstrainedObjective.build(ObjectiveWeights())


@pytest.fixture()
def run_dir(tmp_path: Path) -> RunPaths:
    return run_paths("pytest-smoke", tmp_path / "artifacts").create()


def tiny_overrides(**train: Any) -> ExperimentConfig:
    """Rebuild the smoke configuration with individual training values replaced."""
    raw: dict[str, Any] = {
        "cohort": {
            "n_records": 160,
            "n_silent": 40,
            "records_per_patient": 2,
            "prevalence": 0.194,
            "prevalence_ci": [0.182, 0.206],
            "sites": ["A", "B", "C", "D", "E", "F", "G"],
            "development_sites": ["A", "B", "C", "D", "E"],
            "holdout_sites": ["F", "G"],
            "regions": ["Region I", "Region II", "Region III", "Region IV"],
        },
        "supervision": {"fraction": 0.5, "block_coherence": 0.5, "noise": 0.4},
        "vocabulary": {
            "size": 48,
            "blocks": [
                list(range(0, 12)),
                list(range(12, 20)),
                list(range(20, 28)),
                list(range(28, 34)),
                list(range(34, 42)),
                list(range(42, 48)),
            ],
            "programme_ids": ["P1", "P2", "P3", "P4", "P5", "P6"],
        },
        "model": {
            "streams": {
                "ct": {"input_shape": [1, 2, 4, 4], "width": 8, "depth": 1},
                "mri": {"input_shape": [1, 2, 4, 4], "width": 8, "depth": 1},
                "histology": {"input_shape": [16], "width": 8, "depth": 1},
                "labs": {"input_shape": [6], "width": 8, "depth": 1},
            },
            "fused_dim": 16,
            "representation_dim": 16,
            "probe_hidden": 16,
            "interval_hidden": 16,
            "decoder_hidden": 16,
            "site_classifier_hidden": 16,
        },
        "objective": {"invariance_mode": "variance"},
        "train": {
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "batch_size": 16,
            "grad_accum": 1,
            "world_size": 1,
            "lr_head": 0.01,
            "lr_encoder": 0.001,
            "weight_decay": 0.0001,
            "warmup_fraction": 0.1,
            "grad_clip": 1.0,
            "precision": "fp32",
            "patience": 2,
            "seed": 7,
            "ensemble_members": 1,
            "modality_dropout": 0.2,
            "discrimination_floor": 0.4,
            "equivalence_margin": 0.015,
            "null_band": 0.003,
            "multiplicity_family": 5,
            "interval_nominal_coverage": 0.90,
        },
        "arm": "constrained",
    }
    block = raw["train"]
    if not isinstance(block, dict):
        raise TypeError("the fixture lost its train block")
    block.update(train)
    return parse_config(raw, "_smoke-test")
