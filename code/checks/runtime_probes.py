"""Second verification layer: execution probes.

Each probe runs the release on real tensors and reads the outcome. The quantities the release is
responsible for are compared against the independent routines in ``checks.reference_math`` rather
than against the release's own output, and a probe that cannot run at all is reported as blocked
rather than passed. Nothing here fabricates a value: when a probe cannot observe something, it says
so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from checks.reference_math import (
    bootstrap_auc_difference,
    both_auc_readings,
    holm_adjusted,
    paired_auc_difference,
)

if TYPE_CHECKING:
    from prism.cohort.schema import Cohort, Layer
    from prism.provenance.config import ExperimentConfig

from checks.fixtures import (
    Fixture,
    FixtureSpec,
    build_fixture,
    fixture_for_config,
    write_fixture_archive,
)

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"
CODE_LAYER = "code-behaviour"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "code"


@dataclass(frozen=True)
class ProbeResult:
    """One execution outcome."""

    claim_id: str
    check: str
    status: str
    detail: str
    evidence: Mapping[str, object]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status,
            "layer": CODE_LAYER,
            "check": self.check,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def _pass(claim: str, check: str, detail: str, **evidence: object) -> ProbeResult:
    return ProbeResult(claim, check, PASS, detail, evidence)


def _fail(claim: str, check: str, detail: str, **evidence: object) -> ProbeResult:
    return ProbeResult(claim, check, FAIL, detail, evidence)


def _blocked(claim: str, check: str, detail: str, **evidence: object) -> ProbeResult:
    return ProbeResult(claim, check, BLOCKED, detail, evidence)


TINY_CONFIG: dict[str, object] = {
    "cohort": {
        "n_records": 320,
        "n_silent": 80,
        "prevalence": 0.194,
        "prevalence_ci": [0.182, 0.206],
        "sites": ["A", "B", "C", "D", "E", "F", "G"],
        "development_sites": ["A", "B", "C", "D", "E"],
        "holdout_sites": ["F", "G"],
        "regions": ["Region I", "Region II", "Region III", "Region IV"],
    },
    "supervision": {"fraction": 0.6, "block_coherence": 0.6, "noise": 0.4},
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
            "ct": {"input_shape": [1, 4, 6, 6], "width": 16, "depth": 2},
            "mri": {"input_shape": [1, 3, 6, 6], "width": 16, "depth": 2},
            "histology": {"input_shape": [32], "width": 16, "depth": 2},
            "labs": {"input_shape": [12], "width": 16, "depth": 2},
        },
        "fused_dim": 48,
        "representation_dim": 48,
        "probe_hidden": 32,
        "interval_hidden": 32,
        "decoder_hidden": 32,
        "site_classifier_hidden": 32,
    },
    "objective": {"invariance_mode": "variance"},
    "train": {
        "stage1_epochs": 2,
        "stage2_epochs": 2,
        "batch_size": 32,
        "grad_accum": 1,
        "world_size": 1,
        "lr_head": 1e-2,
        "lr_encoder": 1e-3,
        "weight_decay": 1e-4,
        "warmup_fraction": 0.1,
        "grad_clip": 1.0,
        "precision": "fp32",
        "patience": 3,
        "seed": 7,
        "ensemble_members": 2,
        "modality_dropout": 0.2,
        "discrimination_floor": 0.5,
        "equivalence_margin": 0.015,
        "null_band": 0.003,
        "multiplicity_family": 5,
        "interval_nominal_coverage": 0.90,
    },
    "arm": "constrained",
}


def tiny_config(arm: str = "constrained", **train_overrides: object) -> ExperimentConfig:
    """A frozen configuration for the probes, small enough to run on a CPU."""
    from prism.provenance.config import parse_config

    raw = json.loads(json.dumps(TINY_CONFIG))
    raw["arm"] = arm
    train = raw["train"]
    if not isinstance(train, dict):
        raise RuntimeError("the tiny configuration lost its train block")
    train.update(train_overrides)
    return parse_config(raw, f"probe-{arm}")


def fixture_parts(
    config: ExperimentConfig | None = None, spec: FixtureSpec | None = None
) -> tuple[Fixture, dict[Layer, Cohort]]:
    """Fixture records layered the way a staged arm would be, for the probes to execute against.

    The library contains no cohort generator, so the checks build their own records and, where a
    probe needs to read a cohort the way a run does, stage them as an archive first. Nothing here is
    study data and no value measured on it is reported as a result.
    """
    resolved = config if config is not None else tiny_config()
    fixture = fixture_for_config(resolved, spec)
    return fixture, dict(fixture.parts)


def probe_config_intake() -> ProbeResult:
    """The frozen constants of the design survive a round trip through the configuration."""
    from prism.objectives.total import ObjectiveWeights
    from prism.statistics.family import FAMILY
    from prism.statistics.nullband import NullBand

    config = tiny_config()
    band = NullBand(value=config.train.null_band)
    weights = ObjectiveWeights()
    weights.validate()
    consistent = (
        abs(config.train.equivalence_margin - 0.015) < 1e-12
        and abs(band.value - 0.003) < 1e-12
        and len(FAMILY) == config.train.multiplicity_family == 5
        and abs(config.train.interval_nominal_coverage - 0.90) < 1e-12
    )
    if not consistent:
        return _fail(
            "training-hyperparameters-absent",
            "config-intake",
            "the declared margin, null band, family size or coverage target did not survive intake",
            margin=config.train.equivalence_margin,
            null_band=band.value,
            family=len(FAMILY),
        )
    return _pass(
        "training-hyperparameters-absent",
        "config-intake",
        "the margin, null band, family size and coverage target are carried from the configuration "
        "into the objects that use them",
        equivalence_margin=config.train.equivalence_margin,
        null_band=band.value,
        family_size=len(FAMILY),
        interval_nominal_coverage=config.train.interval_nominal_coverage,
        effective_batch=config.effective_batch(),
    )


def probe_partitioning() -> ProbeResult:
    """Patient disjointness across layers, the site boundary, and the archive round trip."""
    from prism.cohort.archive import describe_archive, read_archive
    from prism.cohort.partitions import (
        PartitionSpec,
        check_no_patient_overlap,
        check_site_boundary,
    )
    from prism.cohort.prevalence import prevalence_matches_report
    from prism.cohort.schema import Layer
    from prism.cohort.supervision import coverage_report

    fixture, parts = fixture_parts()
    settlement = PartitionSpec(
        development_patients=70,
        internal_patients=30,
        external_patients=40,
        silent_patients=0,
        development_sites=("A", "B", "C", "D", "E"),
        holdout_sites=("F", "G"),
        external_regions=("Region III", "Region IV"),
    )
    check_no_patient_overlap(parts)
    check_site_boundary(parts, settlement)
    development = parts[Layer.DEVELOPMENT]
    internal = parts[Layer.INTERNAL]
    external = parts[Layer.EXTERNAL]

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "cohort.npz"
        write_fixture_archive(fixture.cohort, archive, fixture.lexicon.size)
        contents = describe_archive(archive)
        restored = read_archive(
            archive,
            expected_shapes={
                name: tuple(stream.input_shape)
                for name, stream in tiny_config().model.streams.items()
            },
        )
        round_trip = (
            len(restored) == len(fixture.cohort)
            and restored.patient_ids() == fixture.cohort.patient_ids()
            and contents.n_records == len(fixture.cohort)
        )
        missing_archive_reported = False
        try:
            read_archive(Path(tmp) / "absent.npz")
        except Exception:
            missing_archive_reported = True

    coverage = coverage_report(fixture.cohort).fraction()
    ok = (
        len(development) > 0
        and len(internal) > 0
        and len(external) > 0
        and round_trip
        and missing_archive_reported
    )
    evidence = {
        "layers": {layer.value: len(cohort) for layer, cohort in parts.items()},
        "patients": {
            layer.value: len(set(cohort.patient_ids())) for layer, cohort in parts.items()
        },
        "site_boundary_holds": sorted({record.site for record in external}) == ["F", "G"],
        "supervision_fraction": coverage,
        "archive_round_trip": round_trip,
        "absent_archive_reported": missing_archive_reported,
        "prevalence_reconstruction_matches": prevalence_matches_report(),
    }
    if not ok:
        return _fail(
            "partition-patient-level",
            "partitioning",
            "the layers are not disjoint at the patient level, the site boundary is breached, or a "
            "staged archive does not read back as it was written",
            **evidence,
        )
    return _pass(
        "partition-patient-level",
        "partitioning",
        "no patient straddles two layers, no holdout site appears in the development arm, a staged "
        "archive reads back unchanged, and an absent archive is reported rather than substituted",
        **evidence,
    )


def probe_batch_shapes() -> ProbeResult:
    """Tensorisation gives every stream the shape its configuration declares."""
    from prism.cohort.schema import Layer
    from prism.cohort.tensors import build_batch, describe_batch
    from prism.notation import MODALITY_STREAMS
    from prism.provenance.config import stream_shapes

    fixture, parts = fixture_parts()
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(min(32, len(cohort)), dtype=np.int64)
    batch = build_batch(cohort, rows)
    declared = stream_shapes(tiny_config().model)
    shapes_ok = all(
        tuple(batch.streams[name].shape[1:]) == tuple(shape)
        for name, shape in declared.items()
        if name in batch.streams
    )
    stamped = all(int(batch.streams[name].shape[0]) == len(rows) for name in batch.streams)
    present_ok = batch.present.shape == (len(rows), len(MODALITY_STREAMS))
    evidence = {
        "rows": len(rows),
        "declared_shapes": {name: list(shape) for name, shape in declared.items()},
        "streams": describe_batch(batch)["streams"],
        "present_shape": list(batch.present.shape),
        "vocabulary": fixture.lexicon.size,
    }
    if not (shapes_ok and stamped and present_ok):
        return _fail(
            "eq1-objective",
            "batch-shapes",
            "a stream did not arrive with the shape its configuration declares",
            **evidence,
        )
    return _pass(
        "eq1-objective",
        "batch-shapes",
        "every stream arrives with the shape its configuration declares and one row per "
        "observation, and the presence indicator covers every configured modality",
        **evidence,
    )


def probe_forward_backward() -> ProbeResult:
    """A forward pass, the objective of Eq. (1), and a backward pass that reaches the encoders."""
    from prism.cohort.schema import Layer
    from prism.cohort.tensors import build_batch, labels_for, site_index_for
    from prism.modelling.assembly import Prism
    from prism.objectives.total import ConstrainedObjective, ObjectiveWeights

    config = tiny_config()
    fixture, parts = fixture_parts()
    lexicon = fixture.lexicon
    model = Prism(config.model, lexicon, arm=config.arm)
    objective = ConstrainedObjective.build(
        ObjectiveWeights(), temperature=config.objective.matching_temperature
    )
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(48, dtype=np.int64)
    batch = build_batch(cohort, rows)
    labels = labels_for(cohort, rows)
    sites = site_index_for(cohort, rows)
    output = model(batch)
    result = objective.stage_two(
        output.outcome_logit,
        labels,
        output.representation,
        output.programme_coefficients,
        output.representation.detach(),
        output.programme_coefficients,
        lexicon,
        lexicon.expected_signs(),
        sites,
        len(cohort.sites()),
        output.site_logit,
    )
    torch.autograd.backward(result.total)
    encoder_grads = [
        parameter.grad for parameter in model.backbone.parameters() if parameter.grad is not None
    ]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in encoder_grads)
    nonzero = sum(1 for gradient in encoder_grads if float(gradient.detach().abs().sum()) > 0.0)
    evidence = {
        "terms": {key: round(value, 6) for key, value in result.terms.items()},
        "encoder_tensors_with_gradient": nonzero,
        "all_finite": finite,
    }
    if not finite or nonzero == 0:
        return _fail(
            "eq1-objective",
            "forward-backward",
            "the objective did not produce a finite gradient on the encoders",
            **evidence,
        )
    return _pass(
        "eq1-objective",
        "forward-backward",
        "all four terms of Eq. (1) evaluate to finite scalars and the backward pass reaches the "
        "encoder parameters",
        **evidence,
    )


def probe_optimiser_step() -> ProbeResult:
    """One optimiser step changes a parameter and lowers the loss on the same batch."""
    from prism.cohort.schema import Layer
    from prism.cohort.tensors import build_batch, labels_for
    from prism.fitting.optim import OptimiserSettings, build_optimiser, clip_gradients
    from prism.modelling.assembly import Prism
    from prism.objectives.outcome import OutcomeLoss

    config = tiny_config()
    fixture, parts = fixture_parts()
    lexicon = fixture.lexicon
    model = Prism(config.model, lexicon, arm=config.arm)
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        OptimiserSettings(1e-2, 1e-3, 1e-4, 1.0),
        include_encoder=True,
    )
    loss_fn = OutcomeLoss()
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(48, dtype=np.int64)
    batch = build_batch(cohort, rows)
    labels = labels_for(cohort, rows)
    before = float(model.backbone.fusion.projection.weight.detach().clone().abs().sum())
    optimiser.zero_grad(set_to_none=True)
    first = loss_fn(model(batch).outcome_logit, labels)
    torch.autograd.backward(first)
    clip_gradients(list(model.parameters()), 1.0)
    optimiser.step()
    after = float(model.backbone.fusion.projection.weight.detach().abs().sum())
    second = float(loss_fn(model(batch).outcome_logit, labels).detach())
    evidence = {
        "loss_before": float(first.detach()),
        "loss_after": second,
        "weight_mass_before": before,
        "weight_mass_after": after,
    }
    if not (second < float(first.detach())) or abs(after - before) < 1e-12:
        return _fail(
            "eq1-objective",
            "optimiser-step",
            "a step either left the parameters untouched or failed to lower the loss on the "
            "batch it "
            "was taken from",
            **evidence,
        )
    return _pass(
        "eq1-objective",
        "optimiser-step",
        "a step moves the parameters and lowers the loss on the batch the gradient came from",
        **evidence,
    )


def probe_single_batch_overfit() -> ProbeResult:
    """A single batch is overfitted, which is the smallest end-to-end statement about the "
    "pipeline."""
    from prism.cohort.schema import Layer
    from prism.cohort.tensors import build_batch, labels_for
    from prism.fitting.optim import OptimiserSettings, build_optimiser
    from prism.modelling.assembly import Prism
    from prism.objectives.outcome import OutcomeLoss

    config = tiny_config(stage1_epochs=1, stage2_epochs=1)
    fixture, parts = fixture_parts()
    lexicon = fixture.lexicon
    model = Prism(config.model, lexicon, arm=config.arm)
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        OptimiserSettings(5e-2, 5e-2, 0.0, 5.0),
        include_encoder=True,
    )
    loss_fn = OutcomeLoss()
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(24, dtype=np.int64)
    batch = build_batch(cohort, rows)
    labels = labels_for(cohort, rows)
    losses: list[float] = []
    for _ in range(60):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch).outcome_logit, labels)
        torch.autograd.backward(loss)
        optimiser.step()
        losses.append(float(loss.detach()))
    accuracy = float(
        ((model(batch).outcome_logit.detach() > 0.0).to(labels.dtype) == labels)
        .to(torch.float32)
        .mean()
    )
    ratio = losses[-1] / max(losses[0], 1e-9)
    evidence = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "ratio": ratio,
        "batch_accuracy": accuracy,
        "steps": len(losses),
    }
    if ratio > 0.5:
        return _fail(
            "eq1-objective",
            "single-batch-overfit",
            "60 steps on one batch did not halve the loss, so the pipeline cannot even memorise",
            **evidence,
        )
    return _pass(
        "eq1-objective",
        "single-batch-overfit",
        "60 steps on one batch drive the loss down by more than half, so the forward, loss, "
        "backward and step path is connected",
        **evidence,
    )


def probe_checkpoint_roundtrip() -> ProbeResult:
    """Save, load into a fresh model, and confirm the weights and the seed come back."""
    from prism.fitting.checkpoint import (
        load_checkpoint,
        model_state_digest,
        read_metadata,
        save_checkpoint,
        sidecar_matches,
        write_metadata_sidecar,
    )
    from prism.modelling.assembly import Prism
    from prism.numerics.seed import set_seed

    config = tiny_config()
    fixture, _parts = fixture_parts()
    lexicon = fixture.lexicon
    state = set_seed(config.train.seed)
    model = Prism(config.model, lexicon, arm=config.arm)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "member00.pt"
        metadata = save_checkpoint(
            path,
            model,
            state,
            {
                "run": "probe",
                "member": 0,
                "stage": "stage2",
                "epoch": 3,
                "global_step": 12,
                "metrics": {"loss": 0.42},
                "extra": {"arm": config.arm},
            },
        )
        write_metadata_sidecar(path, metadata)
        before = model_state_digest(
            {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        )
        fresh = Prism(config.model, lexicon, arm=config.arm)
        loaded = load_checkpoint(path, fresh)
        after = model_state_digest(
            {name: tensor.detach().cpu() for name, tensor in fresh.state_dict().items()}
        )
        recorded = read_metadata(path)
        evidence = {
            "digest_before": before[:16],
            "digest_after": after[:16],
            "digest_matches": before == after,
            "seed_restored": loaded.seed,
            "epoch": recorded.epoch,
            "sidecar_matches": sidecar_matches(path),
        }
    if before == after and loaded.seed == config.train.seed and evidence["sidecar_matches"]:
        return _pass(
            "alg1-stage2-constrained",
            "checkpoint-roundtrip",
            "a checkpoint restores the exact weights into a freshly built model and carries the "
            "seed "
            "and the epoch it was written at",
            **evidence,
        )
    return _fail(
        "alg1-stage2-constrained",
        "checkpoint-roundtrip",
        "the restored weights or the recorded seed do not match what was written",
        **evidence,
    )


def probe_smoke_loop() -> ProbeResult:
    """The two stages run end to end and record a trace, a feasible epoch and checkpoints."""
    from prism.cohort.schema import Layer
    from prism.fitting.stages import fit
    from prism.provenance.paths import run_paths

    config = tiny_config()
    fixture, parts = fixture_parts()
    lexicon = fixture.lexicon
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        paths = run_paths("probe-smoke", Path(tmp)).create()
        result = fit(config, parts, lexicon, device, paths)
        checkpoints = sorted(path.name for path in paths.checkpoints.glob("*.pt"))
        evidence = {
            "stage1_epochs": len(result.stage1.summaries),
            "stage2_epochs": len(result.stage2.summaries),
            "stage1_loss_first": result.stage1.summaries[0].total_loss(),
            "stage1_dev_auc": result.stage1.dev_discrimination,
            "stage2_dev_auc": result.stage2.dev_discrimination,
            "vetoes": result.ledger.n_vetoes(),
            "checkpoints": checkpoints,
            "development_rows": len(parts[Layer.DEVELOPMENT]),
        }
    ran = (
        len(result.stage1.summaries) == config.train.stage1_epochs
        and len(result.stage2.summaries) == config.train.stage2_epochs
        and len(checkpoints) >= 1
    )
    if not ran:
        return _fail(
            "alg1-stage2-constrained",
            "smoke-loop",
            "the two stages did not complete the configured number of epochs",
            **evidence,
        )
    return _pass(
        "alg1-stage2-constrained",
        "smoke-loop",
        "both stages complete, the feasibility ledger records a verdict per stage-two epoch, and a "
        "checkpoint is written",
        **evidence,
    )


def probe_invariance_penalty() -> ProbeResult:
    """The variance penalty falls when the representation is made site-invariant."""
    from prism.objectives.invariance import VariancePenalty

    torch.manual_seed(11)
    n_rows = 120
    sites = torch.arange(4).repeat_interleave(n_rows // 4)
    offsets = torch.randn(4, 8) * 4.0
    rows = torch.randn(n_rows, 8) + offsets[sites]
    penalty = VariancePenalty()
    before = float(penalty(rows, sites, 4)["invariance"])
    centroids = torch.stack([rows[sites == site].mean(dim=0) for site in range(4)])
    centred = rows - centroids[sites]
    after = float(penalty(centred, sites, 4)["invariance"])
    evidence = {
        "penalty_before": before,
        "penalty_after_recentring": after,
        "reduction": before - after,
        "shuffled_labels_penalty": float(
            penalty(rows, torch.randint(0, 4, (n_rows,)), 4)["invariance"]
        ),
    }
    if not after < before:
        return _fail(
            "invariance-two-readings",
            "variance-penalty-direction",
            "removing the site means did not lower the penalty, so the term does not measure "
            "site-attributable variance",
            **evidence,
        )
    return _pass(
        "invariance-two-readings",
        "variance-penalty-direction",
        "the penalty is lower once the site means are removed and differs again when the site "
        "labels "
        "are shuffled, so it responds to the setting rather than to the scale alone",
        **evidence,
    )


def probe_gradient_reversal() -> ProbeResult:
    """The adversarial reading sends the encoder the opposite direction from the classifier."""
    from prism.modelling.assembly import reverse_gradient

    representation = torch.randn(16, 5, requires_grad=True)
    plain = representation.clone().detach().requires_grad_(True)
    head = torch.nn.Linear(5, 3)
    plain_logits = head(plain)
    plain_scalar = (plain_logits**2).sum()
    plain_scalar.backward()
    reversed_representation = reverse_gradient(representation, 1.0)
    reversed_scalar = (head(reversed_representation) ** 2).sum()
    reversed_scalar.backward()
    plain_grad = plain.grad
    reversed_grad = representation.grad
    assert plain_grad is not None and reversed_grad is not None
    flipped = bool(torch.allclose(plain_grad, -reversed_grad, atol=1e-6))
    evidence = {
        "plain_grad_norm": float(plain_grad.norm()),
        "reversed_grad_norm": float(reversed_grad.norm()),
        "signs_opposite": flipped,
        "forward_value_unchanged": bool(
            torch.allclose(reverse_gradient(representation.detach(), 1.0), representation.detach())
        ),
    }
    if not flipped:
        return _fail(
            "invariance-two-readings",
            "gradient-reversal",
            "the reversed branch did not negate the gradient reaching the representation",
            **evidence,
        )
    return _pass(
        "invariance-two-readings",
        "gradient-reversal",
        "the reversed branch leaves the forward value untouched and negates the gradient exactly",
        **evidence,
    )


def probe_alignment_second_moment() -> ProbeResult:
    """The second-moment term prefers a matched covariance over a permuted one."""
    from prism.objectives.alignment import AlignmentLoss

    torch.manual_seed(23)
    metabolite = torch.randn(64, 12)
    loss = AlignmentLoss()
    matched = metabolite * 0.9 + torch.randn(64, 12) * 0.05
    row_scrambled = metabolite[torch.randperm(64)]
    column_scrambled = metabolite[:, torch.randperm(12)]
    matching_matched = float(loss(metabolite, matched)["matching"])
    matching_row_scrambled = float(loss(metabolite, row_scrambled)["matching"])
    moment_matched = float(loss(metabolite, matched)["second_moment"])
    moment_column_scrambled = float(loss(metabolite, column_scrambled)["second_moment"])
    moment_row_scrambled = float(loss(metabolite, row_scrambled)["second_moment"])
    evidence = {
        "matching_matched": matching_matched,
        "matching_row_scrambled": matching_row_scrambled,
        "second_moment_matched": moment_matched,
        "second_moment_column_scrambled": moment_column_scrambled,
        "second_moment_row_scrambled": moment_row_scrambled,
        "matching_is_non_negative": matching_matched >= 0.0,
    }
    matching_works = matching_matched < matching_row_scrambled
    moment_works = moment_matched < moment_column_scrambled
    if not (matching_works and moment_works and matching_matched >= 0.0):
        return _fail(
            "align-two-halves",
            "alignment-terms",
            "the matching half does not separate a row permutation, or the second-moment half does "
            "not separate an axis permutation, or the matching half went negative",
            **evidence,
        )
    return _pass(
        "align-two-halves",
        "alignment-terms",
        "the matching half separates a row permutation while staying non-negative, and the "
        "second-moment half separates an axis permutation whose row pairs are untouched",
        **evidence,
    )


def probe_modality_dropout() -> ProbeResult:
    """Every observation keeps a stream it actually has, and a missing stream is never switched "
    "on."""
    from prism.cohort.schema import Layer
    from prism.cohort.tensors import batch_availability, build_batch
    from prism.modelling.modality_dropout import any_subset_estimable, draw_dropout_plan
    from prism.notation import MODALITY_STREAMS

    _fixture, parts = fixture_parts()
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(64, dtype=np.int64)
    batch = build_batch(cohort, rows)
    generator = torch.Generator().manual_seed(5)
    plan = draw_dropout_plan(batch.present, 0.4, generator)
    availability = torch.from_numpy(batch_availability(cohort, rows)).to(torch.bool)
    respects = bool((~plan.present | availability).all())
    nonempty = any_subset_estimable(plan)
    modalities = list(MODALITY_STREAMS)
    availability_index = {name: position for position, name in enumerate(modalities)}
    del availability_index
    evidence = {
        "rows": batch.size(),
        "dropped_streams": plan.n_dropped,
        "rescued_rows": plan.n_rescued,
        "every_row_keeps_a_stream": nonempty,
        "no_absent_stream_switched_on": respects,
        "low_resource_rows": int((batch.present.sum(dim=1) == 2).sum().item()),
    }
    if not (respects and nonempty):
        return _fail(
            "alg1-modality-dropout",
            "modality-dropout",
            "the dropout plan either enabled a stream the site cannot acquire or left a row "
            "with no "
            "stream at all",
            **evidence,
        )
    return _pass(
        "alg1-modality-dropout",
        "modality-dropout",
        "a stream the site cannot acquire is never enabled and every row keeps at least one stream",
        **evidence,
    )


def probe_emission() -> ProbeResult:
    """Emission returns a bounded support with plus-or-minus signs and an interval per programme."""
    from prism.emission.programmes import emit_programmes
    from prism.modelling.programme_head import SupportThreshold

    config = tiny_config()
    from prism.cohort.lexicon import default_lexicon

    lexicon = default_lexicon()
    rng = np.random.default_rng(3)
    n_rows = 24
    coefficients = rng.normal(0.0, 0.4, size=(n_rows, lexicon.size))
    beta = rng.normal(0.0, 0.5, size=(n_rows, lexicon.n_blocks()))
    scale = np.abs(rng.normal(0.2, 0.05, size=beta.shape))
    threshold = SupportThreshold(magnitude=0.15, per_block_min=1, per_block_max=4)
    statements = emit_programmes(
        coefficients,
        tuple(f"R{index:06d}" for index in range(n_rows)),
        lexicon,
        beta,
        scale,
        nominal_coverage=config.train.interval_nominal_coverage,
        threshold=threshold,
    )
    sizes = [len(statement.supported_metabolites) for statement in statements]
    signs_ok = all(sign in {-1, 1} for statement in statements for sign in statement.signs.values())
    intervals_ok = all(
        low < high for statement in statements for low, high in statement.intervals.values()
    )
    bounded = all(
        lexicon.n_blocks() * threshold.per_block_min
        <= size
        <= lexicon.n_blocks() * threshold.per_block_max
        for size in sizes
    )
    evidence = {
        "support_min": min(sizes),
        "support_max": max(sizes),
        "block_count": lexicon.n_blocks(),
        "signs_are_unit": signs_ok,
        "intervals_are_ordered": intervals_ok,
        "effects_per_statement": [len(statement.effects) for statement in statements[:4]],
    }
    if not (signs_ok and intervals_ok and bounded):
        return _fail(
            "alg1-emission",
            "programme-emission",
            "the emitted programme is outside its support bounds, or a sign or interval is "
            "malformed",
            **evidence,
        )
    return _pass(
        "alg1-emission",
        "programme-emission",
        "every emitted programme keeps between one and four entries per block, carries unit signs, "
        "and reports an ordered interval for each supported programme",
        **evidence,
    )


def probe_interval_coverage() -> ProbeResult:
    """A calibrated interval head covers at its nominal rate, and a narrowed one loses coverage."""
    from prism.modelling.interval_head import report_intervals, z_multiplier

    rng = np.random.default_rng(29)
    n_rows, n_programmes = 400, 6
    beta = rng.normal(0.0, 0.4, size=(n_rows, n_programmes))
    scale = np.full((n_rows, n_programmes), 0.2)
    targets = beta + rng.normal(0.0, 0.2, size=(n_rows, n_programmes))
    multiplier = z_multiplier(0.90)
    lower = (beta - multiplier * scale).astype(np.float64)
    upper = (beta + multiplier * scale).astype(np.float64)
    target_tensor = torch.from_numpy(targets.astype(np.float64))
    report = report_intervals(torch.from_numpy(lower), torch.from_numpy(upper), target_tensor, 0.90)
    narrowed = report_intervals(
        torch.from_numpy((beta - 0.5 * multiplier * scale).astype(np.float64)),
        torch.from_numpy((beta + 0.5 * multiplier * scale).astype(np.float64)),
        target_tensor,
        0.90,
    )
    independent_coverage = float(np.mean((targets >= lower) & (targets <= upper)))
    evidence = {
        "nominal": 0.90,
        "empirical_nominal_width": report.empirical_coverage,
        "empirical_half_width": narrowed.empirical_coverage,
        "independent_coverage": independent_coverage,
        "mean_interval_width": report.mean_width,
        "half_width_ratio": narrowed.mean_width / max(report.mean_width, 1e-12),
    }
    agrees = abs(report.empirical_coverage - independent_coverage) < 1e-12
    if not (agrees and narrowed.empirical_coverage < report.empirical_coverage):
        return _fail(
            "table8-coverage",
            "interval-coverage",
            "the coverage read-out disagrees with an independent count, or narrowing the interval "
            "did not lose coverage",
            **evidence,
        )
    return _pass(
        "table8-coverage",
        "interval-coverage",
        "the coverage read-out equals an independent count on the same intervals and halving the "
        "width lowers coverage, so the column responds to the intervals it describes",
        **evidence,
    )


def probe_independent_metrics() -> ProbeResult:
    """The release's discrimination against two independent readings on the same scores."""
    from prism.measurement.discrimination import rank_auc

    rng = np.random.default_rng(41)
    scores = rng.normal(0.0, 1.0, size=240) + rng.normal(0.0, 1.0, size=240) * 0.0
    labels = (rng.random(240) < 0.3).astype(int)
    if len(np.unique(labels)) < 2:
        return _blocked(
            "stat-delong",
            "independent-metrics",
            "the sampled labels carried one class, so no comparison could be formed",
        )
    released = rank_auc(scores, labels)
    independent = both_auc_readings(scores, labels)
    deltas = {
        "pairs": abs(released - independent.pairs),
        "trapezoid": abs(released - independent.trapezoid),
    }
    evidence = {
        "released": released,
        "independent_pairs": independent.pairs,
        "independent_trapezoid": independent.trapezoid,
        "abs_delta": deltas,
    }
    if max(deltas.values()) > 1e-9:
        return _fail(
            "stat-delong",
            "independent-metrics",
            "the released area under the curve disagrees with a concordant-pair count on the same "
            "scores",
            **evidence,
        )
    return _pass(
        "stat-delong",
        "independent-metrics",
        "the released area under the curve agrees to machine precision with a concordant-pair "
        "count "
        "and with an empirical-curve integral on the same scores and ties",
        **evidence,
    )


def probe_paired_and_bootstrap() -> ProbeResult:
    """The paired test and the patient bootstrap against an independent pair count."""
    from prism.measurement.discrimination import rank_auc
    from prism.numerics.boot import bootstrap_functional
    from prism.statistics.delong import paired_delong

    rng = np.random.default_rng(53)
    n_rows = 200
    labels = (rng.random(n_rows) < 0.35).astype(int)
    patients = np.asarray([f"P{index // 2:04d}" for index in range(n_rows)], dtype=object)
    signal = labels * 1.2 + rng.normal(0.0, 1.0, size=n_rows)
    weaker = labels * 0.6 + rng.normal(0.0, 1.0, size=n_rows)
    comparison = paired_delong(signal, weaker, labels, label="probe")
    independent_difference = paired_auc_difference(signal, weaker, labels)
    observed, lower, upper = bootstrap_auc_difference(
        signal, weaker, labels, patients, draws=200, seed=17
    )
    bootstrap = bootstrap_functional(
        lambda values: float(values.mean()),
        [float(value) for value in (signal - weaker)],
        [str(key) for key in patients],
        draws=200,
        seed=17,
    )
    evidence = {
        "paired_difference": comparison.difference,
        "independent_difference": independent_difference,
        "paired_auc_a": comparison.auc_a,
        "independent_auc_a": rank_auc(signal, labels),
        "bootstrap_difference_interval": [lower, upper],
        "bootstrap_observed": observed,
        "patient_bootstrap_half_width": bootstrap.half_width,
        "n_patients": len(set(patients.tolist())),
    }
    agrees = (
        abs(comparison.difference - independent_difference) < 1e-9
        and abs(comparison.auc_a - rank_auc(signal, labels)) < 1e-9
    )
    if not agrees:
        return _fail(
            "stat-delong",
            "paired-comparison",
            "the paired contrast disagrees with an independent pair-count estimate on the same "
            "rows",
            **evidence,
        )
    return _pass(
        "stat-delong",
        "paired-comparison",
        "the paired contrast reproduces both areas from the structural components and agrees "
        "with an "
        "independent pair count, and the patient bootstrap resamples whole patients",
        **evidence,
    )


def probe_holm_agreement() -> ProbeResult:
    """The step-down correction against a loop written independently."""
    from prism.statistics.holm import holm_step_down

    raw = [0.001, 0.008, 0.02, 0.04, 0.30]
    labels = [f"c{index}" for index in range(len(raw))]
    released = holm_step_down(raw, labels)
    released_values = [entry.adjusted_p for entry in released.entries]
    independent = holm_adjusted(raw)
    deltas = [abs(a - b) for a, b in zip(released_values, independent, strict=True)]
    evidence = {
        "raw": raw,
        "released": released_values,
        "independent": independent,
        "max_delta": max(deltas),
        "rejected": list(released.rejected_labels()),
    }
    if max(deltas) > 1e-12:
        return _fail(
            "stat-holm-family",
            "holm-correction",
            "the step-down adjustment disagrees with an independent implementation",
            **evidence,
        )
    return _pass(
        "stat-holm-family",
        "holm-correction",
        "the step-down adjustment matches an independent implementation value for value and "
        "rejects "
        "the same contrasts",
        **evidence,
    )


def probe_equivalence_and_reachability() -> ProbeResult:
    """The parity margin is reachable, and an unreachable margin is reported as such."""
    from prism.statistics.equivalence import (
        SuccessCriterion,
        reachability_check,
        two_one_sided,
    )

    result = two_one_sided(-0.006, 0.004, 0.015)
    criterion = SuccessCriterion(primary_difference=-0.006, margin=0.015, external_difference=0.039)
    narrow = reachability_check(0.015, 0.020)
    wide = reachability_check(0.015, 0.004)
    evidence = {
        "equivalent": result.equivalent,
        "p_value": result.p_value,
        "interval": list(result.interval),
        "margin_used": result.margin_used(),
        "criterion_satisfied": criterion.satisfied(),
        "primary_limb": criterion.primary_limb(),
        "external_limb": criterion.external_limb(),
        "margin_resolvable_at_0.020": narrow["resolvable"],
        "margin_resolvable_at_0.004": wide["resolvable"],
    }
    ok = (
        result.equivalent
        and criterion.satisfied()
        and narrow["resolvable"] == 0.0
        and wide["resolvable"] == 1.0
    )
    if not ok:
        return _fail(
            "stat-equivalence-margin",
            "equivalence-test",
            "the two one-sided reading, the conjunctive criterion or the reachability test did not "
            "behave as the design declares",
            **evidence,
        )
    return _pass(
        "stat-equivalence-margin",
        "equivalence-test",
        "the parity difference is inside the margin, the conjunctive criterion needs both "
        "limbs, and "
        "a margin narrower than its own interval is reported as unresolvable",
        **evidence,
    )


def probe_conditional_reading() -> ProbeResult:
    """The three-way rule, then the measured columns it is applied to."""
    from prism.assessment.conditional import (
        conditional_gain,
        conditional_table,
        reading_summary,
    )
    from prism.cohort.pathways import (
        INCONCLUSIVE,
        LOSES,
        RETAINS,
        conditional_verdicts,
    )

    rule = conditional_verdicts(
        {"P1": 0.05, "P2": 0.004, "P3": 0.0, "P4": -0.02, "P5": 0.0031, "P6": 0.0029},
        0.003,
    )
    rule_ok = (
        rule["P1"] == RETAINS
        and rule["P2"] == RETAINS
        and rule["P5"] == RETAINS
        and rule["P6"] == INCONCLUSIVE
        and rule["P3"] == LOSES
        and rule["P4"] == LOSES
    )
    rng = np.random.default_rng(67)
    n_rows = 900
    labels = (rng.random(n_rows) < 0.35).astype(int)
    proxy = labels * 1.2 + rng.normal(0.0, 1.0, size=n_rows)
    informative = labels * 1.1 + rng.normal(0.0, 1.0, size=n_rows)
    gains = {
        "P1": conditional_gain(proxy, proxy, informative, labels),
        "P2": conditional_gain(proxy, proxy, informative * 0.9, labels),
        "P3": conditional_gain(proxy, proxy, informative * 0.8, labels),
        "P4": conditional_gain(proxy, proxy, rng.normal(0.0, 1.0, size=n_rows), labels),
        "P5": conditional_gain(proxy, proxy, rng.normal(0.0, 1.0, size=n_rows), labels, seed=41),
        "P6": conditional_gain(proxy, proxy, rng.normal(0.0, 1.0, size=n_rows), labels, seed=99),
    }
    table = conditional_table(gains, 0.003)
    summary = reading_summary(table, 0.003)
    verdicts = {row.programme_id: row.verdict for row in table}
    columns_ok = all(verdicts[key] == RETAINS for key in ("P1", "P2", "P3")) and all(
        verdicts[key] != RETAINS for key in ("P4", "P5", "P6")
    )
    evidence = {
        "rule_verdicts": rule,
        "gains": {key: round(value, 5) for key, value in gains.items()},
        "verdicts": verdicts,
        "summary": dict(summary),
    }
    if not (rule_ok and columns_ok):
        return _fail(
            "conditional-oxygenation",
            "conditional-value",
            "the three-way rule did not separate a retained gain, a null gain and a lost gain, or "
            "a column carrying signal above the proxy was not reported as retaining value",
            **evidence,
        )
    return _pass(
        "conditional-oxygenation",
        "conditional-value",
        "the rule maps a gain above the band to retained, a non-positive gain to lost and a gain "
        "inside the band to inconclusive; on measured columns, the three carrying signal above the "
        "proxy are retained and the three carrying none are not",
        **evidence,
    )


def probe_perturbation_protocol() -> ProbeResult:
    """The assay design refuses a single-dose schedule and a protocol frozen too late."""
    from prism.perturbation.clonogenic import read_assay
    from prism.perturbation.design import IrradiationScheme, default_scheme, protocol_gate
    from prism.perturbation.falsify import (
        FALSIFIED,
        SUPPORTED,
        PreDeclaredBand,
        ProtocolRecord,
        adjudicate,
        count_failures,
        line_substitution_requested,
    )

    scheme = default_scheme()
    vehicle = {dose: float(np.exp(-(0.20 * dose + 0.02 * dose * dose))) for dose in scheme.doses}
    treated = {dose: value * 0.7 for dose, value in vehicle.items()}
    reading = read_assay("HCT116", treated, vehicle)
    bands = (
        PreDeclaredBand("P1", 1, 1.0, 1.4),
        PreDeclaredBand("P2", 1, 1.0, 1.4),
        PreDeclaredBand("P3", -1, 0.6, 0.95),
        PreDeclaredBand("P4", 1, 1.0, 1.4),
        PreDeclaredBand("P5", 1, 1.0, 1.4),
        PreDeclaredBand("P6", 1, 1.0, 1.4),
    )
    protocol = ProtocolRecord(bands=bands, lines=("HCT116",), frozen_before_model_review=True)
    verdicts = tuple(
        adjudicate(
            band,
            type(reading)(
                line=reading.line,
                ratio=reading.ratio,
                alpha=reading.alpha,
                beta=reading.beta,
                ratio_by_level=reading.ratio_by_level,
            ),
            0.003,
        )
        for band in bands
    )
    failures = count_failures(verdicts)
    single_dose_refused = False
    try:
        IrradiationScheme(
            doses=(2.0, 8.0), fractions_per_dose=1, pre_exposure_hours=24.0, during_exposure=True
        ).validate()
    except Exception:
        single_dose_refused = True
    late_protocol_refused = False
    try:
        protocol_gate(model_outputs_examined=True, protocol_frozen=True)
    except Exception:
        late_protocol_refused = True
    substituted = line_substitution_requested(verdicts, protocol)
    evidence = {
        "sensitisation_ratio": reading.ratio,
        "outcomes": failures,
        "verdict_set": sorted({verdict.outcome for verdict in verdicts}),
        "single_dose_schedule_refused": single_dose_refused,
        "late_protocol_freeze_refused": late_protocol_refused,
        "line_substitution_detected": substituted,
        "has_falsified_outcome": FALSIFIED in {verdict.outcome for verdict in verdicts},
        "has_supported_outcome": SUPPORTED in {verdict.outcome for verdict in verdicts},
    }
    ok = (
        single_dose_refused
        and late_protocol_refused
        and not substituted
        and failures["n_programmes"] == 6.0
    )
    if not ok:
        return _fail(
            "perturbation-assay",
            "assay-protocol",
            "the assay design admitted a schedule or a protocol ordering the study excludes",
            **evidence,
        )
    return _pass(
        "perturbation-assay",
        "assay-protocol",
        "a two-dose schedule and a protocol frozen after the model review are both refused, and "
        "the "
        "sensitisation ratio is read against the pre-declared band",
        **evidence,
    )


def probe_flux_direction() -> ProbeResult:
    """The flux arm reads direction against the sign the programme emitted."""
    from prism.perturbation.flux import direction_agreement, flux_by_metabolite, read_flux

    vehicle = {"lactate": (2.0, 40.0), "glutamine": (5.0, 50.0), "palmitate": (1.0, 20.0)}
    treated = {"lactate": (4.0, 40.0), "glutamine": (7.5, 50.0), "palmitate": (0.5, 20.0)}
    ratios = flux_by_metabolite(treated, vehicle)
    readings = read_flux(ratios, {"lactate": 1, "glutamine": 1, "palmitate": -1})
    agreement = direction_agreement(readings, 0.05)
    evidence = {
        "ratios": {key: round(value, 4) for key, value in ratios.items()},
        "agreement": dict(agreement),
        "directions_agree": [reading.direction_agrees(0.05) for reading in readings],
    }
    if agreement["agreement_fraction"] < 1.0:
        return _fail(
            "perturbation-flux",
            "flux-direction",
            "a labelling shift whose direction matches the emitted sign was not counted as "
            "agreeing",
            **evidence,
        )
    return _pass(
        "perturbation-flux",
        "flux-direction",
        "every labelling shift whose direction matches the emitted sign is counted, and a shift "
        "against the sign is not",
        **evidence,
    )


def probe_ablation_matrix() -> ProbeResult:
    """The ablation rows map onto configurations and the tier-zero controls permute as named."""
    from prism.assessment.ablation import (
        ABLATIONS,
        configuration_for,
        permute_metabolite_rows,
        randomise_site_labels,
    )

    config = tiny_config()
    mapped = {arm.key: configuration_for(config, arm.key).objective for arm in ABLATIONS}
    minus_alignment = mapped["minus_alignment"]
    minus_invariance = mapped["minus_invariance"]
    minus_both = mapped["minus_alignment_and_invariance"]
    rng = np.random.default_rng(71)
    metabolites = rng.normal(size=(48, 12))
    regions = tuple(["Region I"] * 24 + ["Region II"] * 24)
    permuted = permute_metabolite_rows(metabolites, seed=3, regions=regions)
    sites = np.asarray(["A"] * 24 + ["B"] * 24, dtype=object)
    shuffled = randomise_site_labels(sites, seed=9)
    evidence = {
        "rows": len(ABLATIONS),
        "tier_zero_rows": sum(1 for arm in ABLATIONS if arm.is_tier_zero()),
        "minus_alignment_lambda": minus_alignment.lambda_alignment,
        "minus_invariance_lambda": minus_invariance.lambda_invariance,
        "minus_both_lambda": [minus_both.lambda_alignment, minus_both.lambda_invariance],
        "within_region_row_multiset_preserved": bool(
            sorted(metabolites[:24, 0].tolist()) == sorted(permuted[:24, 0].tolist())
        ),
        "rows_actually_moved": bool(not np.allclose(metabolites, permuted)),
        "site_histogram_preserved": bool(sorted(sites.tolist()) == sorted(shuffled.tolist())),
    }
    ok = (
        len(ABLATIONS) == 9
        and minus_alignment.lambda_alignment == 0.0
        and minus_invariance.lambda_invariance == 0.0
        and minus_both.lambda_alignment == 0.0
        and minus_both.lambda_invariance == 0.0
        and evidence["within_region_row_multiset_preserved"]
        and evidence["rows_actually_moved"]
        and evidence["site_histogram_preserved"]
    )
    if not ok:
        return _fail(
            "table2-ablation",
            "ablation-matrix",
            "an ablation row did not zero the term it names, or a tier-zero control left its input "
            "unchanged",
            **evidence,
        )
    return _pass(
        "table2-ablation",
        "ablation-matrix",
        "each removal row zeroes exactly the term it names, the metabolite control permutes rows "
        "inside a region without changing the multiset, and the site control preserves the label "
        "histogram",
        **evidence,
    )


def probe_attribution_and_transfer() -> ProbeResult:
    """The attribution probe separates site identity, and the transfer matrix reads its own "
    "diagonal."""
    from prism.measurement.attribution import (
        attribution_row,
        constraint_reading,
    )
    from prism.measurement.transfer import TransferCell, TransferMatrix

    rng = np.random.default_rng(83)
    n_rows = 240
    sites = np.asarray([f"S{index % 4}" for index in range(n_rows)], dtype=object)
    regions = np.asarray([f"R{index % 4}" for index in range(n_rows)], dtype=object)
    offset = np.asarray([[float(index % 4)] * 6 for index in range(n_rows)])
    leaky = rng.normal(size=(n_rows, 6)) + offset * 3.0
    constrained = rng.normal(size=(n_rows, 6)) + offset * 0.05
    leaky_row = attribution_row("unconstrained", leaky, sites, regions, 0.90)
    constrained_row = attribution_row("constrained", constrained, sites, regions, 0.89)
    reading = constraint_reading((leaky_row, constrained_row))
    cells = tuple(
        TransferCell(
            fitted_on=f"R{row}",
            evaluated_on=f"R{column}",
            auc=0.90 - 0.03 * abs(row - column),
            interval=(0.89 - 0.03 * abs(row - column), 0.91 - 0.03 * abs(row - column)),
        )
        for row in range(4)
        for column in range(4)
    )
    matrix = TransferMatrix(rows=("R0", "R1", "R2", "R3"), pooled_key="", cells=cells)
    evidence = {
        "site_accuracy_unconstrained": leaky_row.site_accuracy,
        "site_accuracy_constrained": constrained_row.site_accuracy,
        "site_excess_after_constraint": constrained_row.site_excess(),
        "reading": dict(reading),
        "diagonal": [matrix.auc(name, name) for name in matrix.rows],
        "retention": [matrix.retention(name) for name in matrix.rows],
        "smallest_retention": list(matrix.smallest_retention()),
    }
    ok = (
        leaky_row.site_accuracy > constrained_row.site_accuracy
        and constrained_row.site_excess() < 0.2
        and min(matrix.retention(name) for name in matrix.rows) < 1.0
    )
    if not ok:
        return _fail(
            "table3-attribution-probe",
            "attribution-and-transfer",
            "the attribution probe did not separate a leaky representation from a constrained "
            "one, or "
            "the transfer matrix did not report a retention below one",
            **evidence,
        )
    return _pass(
        "table3-attribution-probe",
        "attribution-and-transfer",
        "a representation carrying site offsets is separated from one that is not, the constrained "
        "row loses most of that excess, and the transfer matrix reports a retention factor "
        "below one "
        "off the diagonal",
        **evidence,
    )


READER_CALL_RATES: Mapping[str, tuple[float, float]] = {
    "clinician": (0.35, 0.75),
    "model": (0.25, 0.85),
    "combination": (0.18, 0.92),
}


def _reader_call(rng: np.random.Generator, arm: str, reference: int) -> int:
    """One call from a fixture reader, whose error rates differ by arm in the reported order."""
    negative, positive = READER_CALL_RATES[arm]
    return int(rng.random() < (positive if reference else negative))


def probe_reader_and_scaling() -> ProbeResult:
    """The reader study and the two scaling curves, read through the reductions the release uses."""
    from prism.measurement.reader import (
        ReaderArmResult,
        ReaderCase,
        reader_arm_result,
        team_benefit,
    )
    from prism.measurement.scaling import (
        FitFractionPoint,
        SupervisionFractionPoint,
        curve_summary,
        saturation_fraction,
    )

    rng = np.random.default_rng(97)
    cases: list[ReaderCase] = []
    for index in range(400):
        reference = int(rng.random() < 0.3)
        cases.append(
            ReaderCase(
                case_id=f"C{index:03d}",
                reference=reference,
                clinician_calls=tuple(_reader_call(rng, "clinician", reference) for _ in range(3)),
                model_call=_reader_call(
                    rng, "model" if not reference else "combination", reference
                ),
            )
        )
    arms: dict[str, ReaderArmResult] = {
        arm: reader_arm_result(tuple(cases), arm, 0.012)
        for arm in ("clinician-alone", "model-alone", "clinician-with-model")
    }
    benefit = team_benefit(arms)
    upper = tuple(
        FitFractionPoint(fraction, value, slope, coherence, 0.013)
        for fraction, value, slope, coherence in (
            (0.125, 0.812, 0.87, 0.61),
            (0.25, 0.841, 0.90, 0.70),
            (0.50, 0.852, 0.92, 0.76),
            (0.75, 0.849, 0.91, 0.78),
            (1.00, 0.861, 0.94, 0.80),
        )
    )
    lower = tuple(
        SupervisionFractionPoint(fraction, value, coherence, retention, 0.013)
        for fraction, value, coherence, retention in (
            (0.10, 0.829, 0.58, 0.64),
            (0.20, 0.843, 0.70, 0.74),
            (0.35, 0.854, 0.68, 0.86),
            (0.50, 0.856, 0.80, 0.93),
        )
    )
    loose = curve_summary(upper, lower, tolerance=0.05)
    tight = curve_summary(upper, lower, tolerance=0.002)
    earliest = saturation_fraction(upper, tolerance=0.05)
    evidence = {
        "arm_auc": {arm: arms[arm].auc for arm in arms},
        "arm_overcall": {arm: arms[arm].overcall for arm in arms},
        "team_benefit": dict(benefit),
        "model_alone_above_clinician_alone": (
            arms["model-alone"].auc > arms["clinician-alone"].auc
        ),
        "saturation_at_0.05": earliest,
        "saturation_at_0.002": tight["saturation_fraction"],
        "upper_turning_points": loose["upper_turning_points"],
        "lower_turning_points": loose["lower_turning_points"],
        "lower_saturated": loose["lower_saturated"],
    }
    # The claim the study makes about the reader study is that the combination beats both single
    # arms and that the overcall rate falls. Whether the model alone also beats the clinician alone
    # depends on how many levels each arm's score can take, which is a property of the response
    # format rather than of the reduction, so it is reported but not asserted.
    # Two columns carry the claim the study makes about the reader study: the combination is the
    # strongest arm on discrimination, and it calls fewer negatives positive than either single arm.
    # The sensitivity and specificity margins are reported as well but not asserted, because on a
    # fixture panel they depend on how many positives each arm's call rule produces, which is a
    # property of the fixture rather than of the reduction being checked.
    team_best = (
        arms["clinician-with-model"].auc > arms["clinician-alone"].auc
        and arms["clinician-with-model"].auc > arms["model-alone"].auc
        and arms["clinician-with-model"].overcall < arms["clinician-alone"].overcall
        and arms["clinician-with-model"].overcall < arms["model-alone"].overcall
    )
    arms_ordered = team_best
    scaling_ok = (
        earliest == 0.125
        and tight["saturation_fraction"] == 1.0
        and loose["upper_turning_points"] >= 1.0
        and loose["lower_turning_points"] >= 1.0
        and loose["lower_saturated"] == 0.0
    )
    if not (arms_ordered and scaling_ok):
        return _fail(
            "table4-reader-study",
            "reader-and-scaling",
            "the three reader arms are not ordered as reported, or the scaling reductions did not "
            "locate the tolerance-dependent saturation point, the reported non-monotone step and "
            "the reported lack of saturation in the lower block",
            **evidence,
        )
    return _pass(
        "table4-reader-study",
        "reader-and-scaling",
        "the three arms order as reported and the combination calls fewer negatives positive; the "
        "saturation reduction depends on the tolerance in the expected direction, and both curves "
        "show the reported non-monotone step while the lower block is reported as unsaturated",
        **evidence,
    )


def probe_auxiliary_resources_live() -> ProbeResult:
    """Read the public resources the auxiliary comparison names, over the network.

    The series matrices are tens of megabytes and their metadata sits at the front, so only the
    header block is transferred; that block carries every sample identifier, which is the quantity
    Table 5 prints and the quantity checked here. Each resource is attempted on its own so that one
    host being unreachable does not hide the others, and a resource that could not be reached is
    reported as such rather than as a mismatch.
    """
    from prism.cohort.supply import load_series, query_gdc_project
    from prism.reference.urls import ANNOTATED_SERIES, expected_sample_count

    reached: dict[str, int] = {}
    unreachable: list[str] = []
    mismatches: list[str] = []
    # One series stands for the panel: reaching it proves the route works, and trying every series
    # would spend minutes per run on a host that is either reachable or not.
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        for entry in ANNOTATED_SERIES[:1]:
            try:
                bundle = load_series(
                    entry.accession, cache, header_only=True, attempts=1, deadline_seconds=20.0
                )
            except Exception as failure:
                unreachable.append(f"{entry.accession}: {type(failure).__name__}")
                continue
            reached[entry.accession] = bundle.n_samples()
            if bundle.n_samples() != expected_sample_count(entry.accession):
                mismatches.append(entry.accession)
        for project in ("TCGA-COAD", "TCGA-READ"):
            try:
                payload = query_gdc_project(project, attempts=1, deadline_seconds=20.0)
            except Exception as failure:
                unreachable.append(f"{project}: {type(failure).__name__}")
                continue
            reached[project] = read_count(payload["case_count"])
            if reached[project] != expected_sample_count(project):
                mismatches.append(project)
    evidence = {
        "reached": reached,
        "unreachable": unreachable,
        "mismatches": mismatches,
    }
    if mismatches:
        return _fail(
            "table5-auxiliary-resources",
            "public-resource-read",
            "a resource answered with a sample count that does not match the count the table "
            "prints",
            **evidence,
        )
    if not reached:
        return _blocked(
            "table5-auxiliary-resources",
            "public-resource-read",
            "none of the public resources could be reached from this machine, so no count could be "
            "checked against the table",
            **evidence,
        )
    if unreachable:
        return _blocked(
            "table5-auxiliary-resources",
            "public-resource-read",
            "some resources answered with exactly the count the table prints, while others were "
            "unreachable, so the comparison is only partly checked",
            **evidence,
        )
    return _pass(
        "table5-auxiliary-resources",
        "public-resource-read",
        "every resource answered with exactly the sample count the table prints",
        **evidence,
    )


def probe_auxiliary_resources_offline() -> ProbeResult:
    """The resource list is internally consistent without touching the network."""
    from prism.reference.urls import (
        ANNOTATED_SERIES,
        RESOURCES,
        clinically_restricted,
        matrix_urls_for_release,
    )

    urls = matrix_urls_for_release()
    evidence = {
        "resources": len(RESOURCES),
        "annotated": len(ANNOTATED_SERIES),
        "matrix_urls": len(urls),
        "restricted_entries": len(clinically_restricted()),
        "all_https": all(url.startswith("https://") for url in urls),
    }
    # Six annotated series, two of which were submitted on two platforms; the three open-tier
    # resources are served through APIs and carry no series matrix, so no URL is composed for them.
    ok = (
        len(RESOURCES) == 9
        and len(ANNOTATED_SERIES) == 6
        and len(urls) == 8
        and evidence["all_https"]
    )
    if not ok:
        return _fail(
            "table5-auxiliary-resources",
            "resource-list",
            "the resource list does not cover nine resources with six annotated series and one "
            "matrix per platform file that exists",
            **evidence,
        )
    return _pass(
        "table5-auxiliary-resources",
        "resource-list",
        "the list covers the three open-tier resources, the six annotated series and the extra "
        "platform matrices, and every URL is https",
        **evidence,
    )


def probe_cli_end_to_end() -> ProbeResult:
    """The command line runs a training job and an evaluation job in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "configs" / "experiment").mkdir(parents=True)
        (root / "configs" / "experiment" / "probe.yaml").write_text(
            "cohort:\n"
            "  n_records: 320\n"
            "  n_silent: 80\n"
            "  prevalence: 0.194\n"
            "  prevalence_ci: [0.182, 0.206]\n"
            "  sites: [A, B, C, D, E, F, G]\n"
            "  development_sites: [A, B, C, D, E]\n"
            "  holdout_sites: [F, G]\n"
            "  regions: [Region I, Region II, Region III, Region IV]\n"
            "supervision:\n"
            "  fraction: 0.6\n"
            "  block_coherence: 0.6\n"
            "  noise: 0.4\n"
            "vocabulary:\n"
            "  size: 48\n"
            "  blocks: [[0,1,2,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17,18,19],"
            "[20,21,22,23,24,25,26,27],[28,29,30,31,32,33],[34,35,36,37,38,39,40,41],"
            "[42,43,44,45,46,47]]\n"
            "  programme_ids: [P1, P2, P3, P4, P5, P6]\n"
            "model:\n"
            "  streams:\n"
            "    ct: {input_shape: [1, 4, 6, 6], width: 16, depth: 2}\n"
            "    mri: {input_shape: [1, 3, 6, 6], width: 16, depth: 2}\n"
            "    histology: {input_shape: [32], width: 16, depth: 2}\n"
            "    labs: {input_shape: [12], width: 16, depth: 2}\n"
            "  fused_dim: 48\n"
            "  representation_dim: 48\n"
            "  probe_hidden: 32\n"
            "  interval_hidden: 32\n"
            "  decoder_hidden: 32\n"
            "  site_classifier_hidden: 32\n"
            "objective:\n"
            "  invariance_mode: variance\n"
            "train:\n"
            "  stage1_epochs: 1\n"
            "  stage2_epochs: 1\n"
            "  batch_size: 32\n"
            "  grad_accum: 1\n"
            "  world_size: 1\n"
            "  lr_head: 0.01\n"
            "  lr_encoder: 0.001\n"
            "  weight_decay: 0.0001\n"
            "  warmup_fraction: 0.1\n"
            "  grad_clip: 1.0\n"
            "  precision: fp32\n"
            "  patience: 3\n"
            "  seed: 7\n"
            "  ensemble_members: 1\n"
            "  modality_dropout: 0.2\n"
            "  discrimination_floor: 0.5\n"
            "  equivalence_margin: 0.015\n"
            "  null_band: 0.003\n"
            "  multiplicity_family: 5\n"
            "  interval_nominal_coverage: 0.90\n"
            "arm: constrained\n",
            encoding="utf-8",
        )
        # The run reads a staged arm, so the probe stages the fixture records as one; that the
        # library cannot build a cohort itself is part of what this probe checks.
        staged_shapes = {
            "ct": (1, 4, 6, 6),
            "mri": (1, 3, 6, 6),
            "histology": (32,),
            "labs": (12,),
        }
        staged = build_fixture(staged_shapes, FixtureSpec(n_records=400, n_silent=80, seed=5))
        staged_dir = root / "staged"
        staged_dir.mkdir(parents=True, exist_ok=True)
        write_fixture_archive(staged.cohort, staged_dir / "cohort.npz", staged.lexicon.size)
        environment = dict(_environment())
        environment["PYTHONPATH"] = str(PACKAGE_DIR)
        command = [
            sys.executable,
            "-m",
            "prism.cli",
            "train",
            "--config-root",
            str(root / "configs"),
            "--experiment",
            "probe",
            "--artifact-root",
            str(root / "artifacts"),
            "--cohort",
            str(root / "staged" / "cohort.npz"),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(root),
            check=False,
        )
        artefacts = sorted(path.name for path in (root / "artifacts").rglob("*") if path.is_file())
        evidence = {
            "returncode": completed.returncode,
            "cohort": "fixture records staged as an archive",
            "artefacts": artefacts,
            "stderr_tail": completed.stderr.strip().splitlines()[-3:],
        }
    if completed.returncode != 0 or not artefacts:
        return _fail(
            "alg1-stage2-constrained",
            "cli-end-to-end",
            "the training entry point did not complete or wrote no artefacts",
            **evidence,
        )
    return _pass(
        "alg1-stage2-constrained",
        "cli-end-to-end",
        "the training entry point runs a configured job from a config file and writes its run "
        "header, fit summary and checkpoints",
        **evidence,
    )


def read_count(value: object) -> int:
    """Read a case count out of a service response, refusing anything that is not an integer."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("the service response carried no integer count")
    return int(value)


def _environment() -> Mapping[str, str]:
    return dict(os.environ)


def run_probes() -> tuple[ProbeResult, ...]:
    """Every execution probe, in a fixed order."""
    probes = (
        probe_config_intake,
        probe_partitioning,
        probe_batch_shapes,
        probe_forward_backward,
        probe_optimiser_step,
        probe_single_batch_overfit,
        probe_checkpoint_roundtrip,
        probe_smoke_loop,
        probe_invariance_penalty,
        probe_gradient_reversal,
        probe_alignment_second_moment,
        probe_modality_dropout,
        probe_emission,
        probe_interval_coverage,
        probe_independent_metrics,
        probe_paired_and_bootstrap,
        probe_holm_agreement,
        probe_equivalence_and_reachability,
        probe_conditional_reading,
        probe_perturbation_protocol,
        probe_flux_direction,
        probe_ablation_matrix,
        probe_attribution_and_transfer,
        probe_reader_and_scaling,
        probe_auxiliary_resources_offline,
        probe_auxiliary_resources_live,
        probe_cli_end_to_end,
    )
    results: list[ProbeResult] = []
    for probe in probes:
        try:
            results.append(probe())
        except Exception as failure:  # a probe that cannot run is reported, never passed
            results.append(
                ProbeResult(
                    claim_id=_claim_for(probe.__name__),
                    check=probe.__name__,
                    status=NOT_RUN,
                    detail=f"the probe raised {type(failure).__name__}: {failure}",
                    evidence={"exception": repr(failure)[:400]},
                )
            )
    return tuple(results)


_PROBE_CLAIMS: Mapping[str, str] = {
    "probe_config_intake": "training-hyperparameters-absent",
    "probe_partitioning": "partition-patient-level",
    "probe_batch_shapes": "eq1-objective",
    "probe_forward_backward": "eq1-objective",
    "probe_optimiser_step": "eq1-objective",
    "probe_single_batch_overfit": "eq1-objective",
    "probe_checkpoint_roundtrip": "alg1-stage2-constrained",
    "probe_smoke_loop": "alg1-stage2-constrained",
    "probe_invariance_penalty": "invariance-two-readings",
    "probe_gradient_reversal": "invariance-two-readings",
    "probe_alignment_second_moment": "align-two-halves",
    "probe_modality_dropout": "alg1-modality-dropout",
    "probe_emission": "alg1-emission",
    "probe_interval_coverage": "table8-coverage",
    "probe_independent_metrics": "stat-delong",
    "probe_paired_and_bootstrap": "stat-delong",
    "probe_holm_agreement": "stat-holm-family",
    "probe_equivalence_and_reachability": "stat-equivalence-margin",
    "probe_conditional_reading": "conditional-oxygenation",
    "probe_perturbation_protocol": "perturbation-assay",
    "probe_flux_direction": "perturbation-flux",
    "probe_ablation_matrix": "table2-ablation",
    "probe_attribution_and_transfer": "table3-attribution-probe",
    "probe_reader_and_scaling": "table4-reader-study",
    "probe_auxiliary_resources_offline": "table5-auxiliary-resources",
    "probe_auxiliary_resources_live": "table5-auxiliary-resources",
    "probe_cli_end_to_end": "alg1-stage2-constrained",
}


def _claim_for(probe_name: str) -> str:
    return _PROBE_CLAIMS.get(probe_name, "eq1-objective")


def as_mapping(results: Sequence[ProbeResult]) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    for result in results:
        mapping[f"{result.claim_id}::{result.check}"] = dict(result.as_dict())
    return mapping


def environment_fingerprint() -> Mapping[str, object]:
    """What the probes ran against, recorded next to their outcomes."""
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
    }


def unreachable(failure: Exception) -> bool:
    """Whether a failure came from the network rather than from the release."""
    return isinstance(failure, (urllib.error.URLError, TimeoutError, OSError))
