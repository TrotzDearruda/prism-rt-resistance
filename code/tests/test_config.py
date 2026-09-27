"""Configuration intake: the frozen design constants and the group merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from prism.errors import ConfigError
from prism.provenance.config import (
    ORIGIN_ENGINEERING_DEFAULT,
    load_experiment,
    parse_config,
)
from prism.statistics.family import FAMILY
from tests.conftest import tiny_overrides


def test_smoke_configuration_loads(smoke_config) -> None:
    assert smoke_config.arm == "constrained"
    assert smoke_config.vocabulary.size == 48
    assert smoke_config.effective_batch() == (
        smoke_config.train.batch_size
        * smoke_config.train.grad_accum
        * smoke_config.train.world_size
    )


def test_reported_design_constants_survive_intake(smoke_config) -> None:
    assert smoke_config.train.equivalence_margin == pytest.approx(0.015)
    assert smoke_config.train.null_band == pytest.approx(0.003)
    assert smoke_config.train.multiplicity_family == len(FAMILY) == 5
    assert smoke_config.train.interval_nominal_coverage == pytest.approx(0.90)
    assert smoke_config.cohort.prevalence == pytest.approx(0.194)
    assert smoke_config.cohort.prevalence_ci == pytest.approx((0.182, 0.206))


def test_every_experiment_file_loads() -> None:
    directory = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    files = sorted(directory.glob("*.yaml"))
    assert len(files) >= 15
    for path in files:
        config = load_experiment(path)
        assert config.name
        assert config.vocabulary.size == 48


def test_ablations_move_only_their_own_weight() -> None:
    directory = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    base = load_experiment(directory / "main.yaml")
    no_alignment = load_experiment(directory / "ablation_minus_alignment.yaml")
    no_invariance = load_experiment(directory / "ablation_minus_invariance.yaml")
    both = load_experiment(directory / "ablation_minus_alignment_and_invariance.yaml")
    assert base.objective.lambda_alignment == pytest.approx(1.0)
    assert base.objective.lambda_invariance == pytest.approx(0.5)
    assert no_alignment.objective.lambda_alignment == 0.0
    assert no_alignment.objective.lambda_invariance == pytest.approx(0.5)
    assert no_invariance.objective.lambda_invariance == 0.0
    assert both.objective.lambda_alignment == 0.0
    assert both.objective.lambda_invariance == 0.0


def test_stream_configuration_records_its_origin() -> None:
    config = tiny_overrides()
    for stream in config.model.streams.values():
        assert stream.origin == ORIGIN_ENGINEERING_DEFAULT


def test_an_overlapping_site_assignment_is_refused() -> None:
    with pytest.raises(ConfigError):
        parse_config(
            {
                "cohort": {
                    "n_records": 10,
                    "n_silent": 2,
                    "prevalence": 0.2,
                    "prevalence_ci": [0.1, 0.3],
                    "sites": ["A", "B"],
                    "development_sites": ["A"],
                    "holdout_sites": ["A"],
                    "regions": ["Region I"],
                },
                "supervision": {"fraction": 0.5, "block_coherence": 0.5, "noise": 0.4},
                "vocabulary": {
                    "size": 2,
                    "blocks": [[0], [1]],
                    "programme_ids": ["P1", "P2"],
                },
                "model": {
                    "streams": {"labs": {"input_shape": [4], "width": 4, "depth": 1}},
                    "fused_dim": 4,
                    "representation_dim": 4,
                    "probe_hidden": 4,
                    "interval_hidden": 4,
                    "decoder_hidden": 4,
                    "site_classifier_hidden": 4,
                },
                "objective": {},
                "train": {
                    "stage1_epochs": 1,
                    "stage2_epochs": 1,
                    "batch_size": 2,
                    "grad_accum": 1,
                    "world_size": 1,
                    "lr_head": 0.01,
                    "lr_encoder": 0.001,
                    "weight_decay": 0.0,
                    "warmup_fraction": 0.0,
                    "grad_clip": 1.0,
                    "precision": "fp32",
                    "patience": 1,
                    "seed": 1,
                    "ensemble_members": 1,
                    "modality_dropout": 0.1,
                    "discrimination_floor": 0.5,
                    "equivalence_margin": 0.015,
                    "null_band": 0.003,
                    "multiplicity_family": 5,
                    "interval_nominal_coverage": 0.9,
                },
            },
            "overlapping",
        )


def test_a_vocabulary_that_is_not_partitioned_is_refused() -> None:
    with pytest.raises(ConfigError):
        parse_config(
            {
                "cohort": {
                    "n_records": 10,
                    "n_silent": 2,
                    "prevalence": 0.2,
                    "prevalence_ci": [0.1, 0.3],
                    "sites": ["A", "B"],
                    "development_sites": ["A"],
                    "holdout_sites": ["B"],
                    "regions": ["Region I"],
                },
                "supervision": {"fraction": 0.5, "block_coherence": 0.5, "noise": 0.4},
                "vocabulary": {
                    "size": 3,
                    "blocks": [[0], [1]],
                    "programme_ids": ["P1", "P2"],
                },
                "model": {
                    "streams": {"labs": {"input_shape": [4], "width": 4, "depth": 1}},
                    "fused_dim": 4,
                    "representation_dim": 4,
                    "probe_hidden": 4,
                    "interval_hidden": 4,
                    "decoder_hidden": 4,
                    "site_classifier_hidden": 4,
                },
                "objective": {},
                "train": {
                    "stage1_epochs": 1,
                    "stage2_epochs": 1,
                    "batch_size": 2,
                    "grad_accum": 1,
                    "world_size": 1,
                    "lr_head": 0.01,
                    "lr_encoder": 0.001,
                    "weight_decay": 0.0,
                    "warmup_fraction": 0.0,
                    "grad_clip": 1.0,
                    "precision": "fp32",
                    "patience": 1,
                    "seed": 1,
                    "ensemble_members": 1,
                    "modality_dropout": 0.1,
                    "discrimination_floor": 0.5,
                    "equivalence_margin": 0.015,
                    "null_band": 0.003,
                    "multiplicity_family": 5,
                    "interval_nominal_coverage": 0.9,
                },
            },
            "unpartitioned",
        )
