"""Checkpoint integrity, the mini-batch engine and the two-stage smoke loop.

Ref: Algorithm 1 -- these tests exercise the path a reported run takes: a batch is drawn with
modality dropout, the three sub-steps are taken, the development split decides early stopping, and
the feasibility floor is checked per epoch. The configuration used here is ``_smoke.yaml``, which is
smaller than the primary configuration so that a test can finish in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.cohort.schema import Layer
from prism.cohort.supervision import AlignmentBatch
from prism.cohort.tensors import build_batch, labels_for
from prism.errors import CheckpointError, ConfigError
from prism.fitting.checkpoint import (
    FORMAT_VERSION,
    load_checkpoint,
    model_state_digest,
    read_metadata,
    save_checkpoint,
    sidecar_matches,
    write_metadata_sidecar,
)
from prism.fitting.loop import (
    BatchSpec,
    apply_dropout,
    evaluate_logits,
    evaluate_probabilities,
    make_generator,
    outcome_step,
    region_draws_for_window,
    run_epoch,
    supervised_positions,
)
from prism.fitting.optim import (
    OptimiserSettings,
    build_optimiser,
    clip_gradients,
    describe_groups,
    gradient_norm,
    resolve_precision,
)
from prism.fitting.schedule import (
    ScheduleSettings,
    attach_scheduler,
    current_rates,
    schedule_factors,
    steps_per_epoch,
    total_steps_for_stages,
    warmup_cosine_factor,
)
from prism.fitting.stages import development_split, fit
from prism.modelling.assembly import Prism
from prism.numerics.seed import set_seed
from prism.objectives.outcome import OutcomeLoss
from prism.objectives.total import ConstrainedObjective, ObjectiveWeights
from tests.conftest import tiny_overrides


def test_a_checkpoint_round_trips_into_a_fresh_model(tmp_path, lexicon) -> None:
    config = tiny_overrides()
    state = set_seed(config.train.seed)
    model = Prism(config.model, lexicon, arm=config.arm)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    path = tmp_path / "member00.pt"
    metadata = save_checkpoint(
        path,
        model,
        state,
        {
            "run": "pytest",
            "member": 0,
            "stage": "stage2",
            "epoch": 4,
            "global_step": 12,
            "metrics": {"loss": 0.42},
        },
    )
    write_metadata_sidecar(path, metadata)
    fresh = Prism(config.model, lexicon, arm=config.arm)
    recorded = model_state_digest(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    )
    loaded = load_checkpoint(path, fresh)
    restored = model_state_digest(
        {name: tensor.detach().cpu() for name, tensor in fresh.state_dict().items()}
    )
    assert restored == recorded
    assert loaded.seed == config.train.seed
    assert loaded.format_version == FORMAT_VERSION
    assert loaded.epoch == 4
    assert read_metadata(path).global_step == 12
    assert sidecar_matches(path)
    assert path.with_suffix(".pt.json").is_file()


def test_a_checkpoint_survives_a_scheduler_and_optimiser_state(tmp_path, lexicon) -> None:
    config = tiny_overrides()
    state = set_seed(config.train.seed)
    model = Prism(config.model, lexicon, arm=config.arm)
    settings = OptimiserSettings(1e-2, 1e-3, 1e-4, 1.0)
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        settings,
        include_encoder=True,
    )
    scheduler = attach_scheduler(optimiser, ScheduleSettings(total_steps=20, warmup_fraction=0.2))
    path = tmp_path / "member00.pt"
    save_checkpoint(
        path,
        model,
        state,
        {"run": "pytest", "member": 0, "stage": "stage1", "epoch": 1, "global_step": 3},
        optimiser=optimiser,
        scheduler=scheduler,
    )
    fresh = Prism(config.model, lexicon, arm=config.arm)
    fresh_optimiser = build_optimiser(
        fresh.head_parameters(),
        list(fresh.backbone.parameters()),
        settings,
        include_encoder=True,
    )
    fresh_scheduler = attach_scheduler(
        fresh_optimiser, ScheduleSettings(total_steps=20, warmup_fraction=0.2)
    )
    load_checkpoint(path, fresh, optimiser=fresh_optimiser, scheduler=fresh_scheduler)
    assert describe_groups(fresh_optimiser)
    assert set(current_rates(fresh_optimiser)) == {"head", "encoder"}


def test_a_checkpoint_that_fails_its_digest_is_refused(tmp_path, lexicon) -> None:
    config = tiny_overrides()
    state = set_seed(1)
    model = Prism(config.model, lexicon, arm=config.arm)
    path = tmp_path / "member00.pt"
    save_checkpoint(path, model, state, {"run": "pytest", "member": 0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    first = next(iter(payload["model_state"]))
    payload["model_state"][first] = payload["model_state"][first] + 1.0
    torch.save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, Prism(config.model, lexicon, arm=config.arm))
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "absent.pt", model)


def test_the_schedule_warmps_then_decays() -> None:
    settings = ScheduleSettings(total_steps=100, warmup_fraction=0.1)
    factors = schedule_factors(settings)
    assert len(factors) == 100
    assert factors[0] < factors[5] < factors[9]
    assert factors[9] == pytest.approx(1.0)
    assert factors[-1] < 0.05
    assert settings.warmup_steps() == 10
    assert warmup_cosine_factor(0, settings) == pytest.approx(0.1)
    assert steps_per_epoch(1000, 32, 2) == 16
    stage1, stage2 = total_steps_for_stages(1000, 32, 2, 5, 30)
    assert stage1 == 5 * 16
    assert stage2 == 30 * 16


def test_the_precision_names_are_resolved_and_refused() -> None:
    assert resolve_precision("fp32", torch.device("cpu")) is torch.float32
    assert resolve_precision("bf16", torch.device("cpu")) is torch.bfloat16
    with pytest.raises(ConfigError):
        resolve_precision("fp16", torch.device("cpu"))
    with pytest.raises(ConfigError):
        resolve_precision("int8", torch.device("cpu"))


def test_one_batch_is_memorised_by_the_heads(smoke_config, parts, lexicon) -> None:
    """The smallest end-to-end statement about the pipeline: it can fit one batch."""
    config = tiny_overrides(stage1_epochs=1, stage2_epochs=1, batch_size=24)
    set_seed(config.train.seed)
    model = Prism(config.model, lexicon, arm=config.arm)
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        OptimiserSettings(5e-2, 5e-2, 0.0, 5.0),
        include_encoder=True,
    )
    loss_function = OutcomeLoss()
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(24, dtype=np.int64)
    batch = build_batch(cohort, rows)
    labels = labels_for(cohort, rows)
    losses: list[float] = []
    for _ in range(60):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_function(model(batch).outcome_logit, labels)
        torch.autograd.backward(loss)
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0]
    assert gradient_norm(list(model.parameters())) >= 0.0
    clip_gradients(list(model.parameters()), 1.0)


def test_an_epoch_takes_its_three_sub_steps(parts, lexicon, objective, smoke_config) -> None:
    config = tiny_overrides()
    set_seed(config.train.seed)
    model = Prism(config.model, lexicon, arm=config.arm)
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        OptimiserSettings(1e-2, 1e-3, 0.0, 1.0),
        include_encoder=True,
    )
    cohort = parts[Layer.DEVELOPMENT]
    rows = np.arange(min(48, len(cohort)), dtype=np.int64)
    spec = BatchSpec(batch_size=16, grad_accum=1, shuffle=True, seed=3)
    summary = run_epoch(
        model,
        objective,
        optimiser,
        cohort,
        rows,
        spec,
        lexicon,
        lexicon.expected_signs(),
        config.train,
        torch.device("cpu"),
        stage="stage2",
        epoch=0,
        generator=make_generator(3, torch.device("cpu")),
        with_alignment=True,
        with_invariance=True,
    )
    assert summary.n_windows > 0
    assert summary.optimiser_steps >= summary.n_windows
    assert "outcome" in summary.means()
    assert summary.grad_norms
    assert summary.dropped_modalities >= 0


def test_a_supervised_window_produces_an_alignment_draw(parts, lexicon) -> None:
    cohort = parts[Layer.DEVELOPMENT].supervised()
    if len(cohort) < 4:
        pytest.skip("the fixture records carry too little supervision")
    rows = np.arange(min(32, len(cohort)), dtype=np.int64)
    positions = supervised_positions(cohort, rows)
    assert positions.size == rows.size
    draw = region_draws_for_window(cohort, rows, lexicon.size)
    assert draw is not None
    assert isinstance(draw, AlignmentBatch)
    assert draw.metabolite.shape[1] == lexicon.size
    assert draw.metabolite.shape[0] == positions.size


def test_the_development_split_keeps_patients_whole(parts) -> None:
    cohort = parts[Layer.DEVELOPMENT]
    split = development_split(cohort, 0.2, seed=5)
    fit_patients = {cohort.records[int(row)].patient_id for row in split.fit_rows}
    dev_patients = {cohort.records[int(row)].patient_id for row in split.dev_rows}
    assert fit_patients.isdisjoint(dev_patients)
    assert split.fit_rows.size + split.dev_rows.size == len(cohort)


def test_the_two_stage_smoke_loop_runs_and_writes_a_checkpoint(run_dir, parts, lexicon) -> None:
    config = tiny_overrides()
    result = fit(config, parts, lexicon, torch.device("cpu"), run_dir)
    assert len(result.stage1.summaries) == config.train.stage1_epochs
    assert len(result.stage2.summaries) == config.train.stage2_epochs
    assert result.stage1.dev_discrimination
    assert result.stage2.dev_discrimination
    assert result.ledger.verdicts
    assert result.checkpoints
    assert sorted(path.name for path in run_dir.checkpoints.glob("*.pt"))
    assert result.seed_state.seed == config.train.seed


def test_the_held_out_partition_is_read_after_fitting(run_dir, parts, lexicon) -> None:
    config = tiny_overrides()
    result = fit(config, parts, lexicon, torch.device("cpu"), run_dir)
    held = parts[Layer.EXTERNAL]
    rows = np.arange(len(held), dtype=np.int64)
    logits = evaluate_logits(result.model, held, rows, 32, torch.device("cpu"))
    probabilities = evaluate_probabilities(result.model, held, rows, 32, torch.device("cpu"))
    assert logits.shape == (len(held),)
    assert bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


def test_modality_dropout_is_applied_inside_the_engine(parts) -> None:
    cohort = parts[Layer.DEVELOPMENT]
    batch = build_batch(cohort, np.arange(24, dtype=np.int64))
    generator = make_generator(7, torch.device("cpu"))
    dropped, n_dropped, n_rescued = apply_dropout(batch, 0.4, generator)
    assert dropped.present.shape == batch.present.shape
    assert n_dropped >= 0
    assert n_rescued >= 0
    assert bool((dropped.present.sum(dim=1) >= 1).all())


def test_repeated_runs_agree_under_the_same_seed(run_dir, parts, lexicon, tmp_path) -> None:
    config = tiny_overrides()
    first = fit(config, parts, lexicon, torch.device("cpu"), run_dir)
    second = fit(config, parts, lexicon, torch.device("cpu"), run_dir)
    assert first.stage1.dev_discrimination == second.stage1.dev_discrimination
    assert first.stage2.dev_discrimination == second.stage2.dev_discrimination
    assert first.checkpoints[-1].state_digest == second.checkpoints[-1].state_digest


def test_the_objective_stage_two_needs_a_matching_alignment_width(lexicon) -> None:
    objective = ConstrainedObjective.build(ObjectiveWeights())
    with pytest.raises(ConfigError):
        objective.stage_two(
            torch.randn(8),
            torch.randint(0, 2, (8,)).to(torch.float32),
            torch.randn(8, 4),
            torch.randn(8, 4),
            torch.randn(8, lexicon.size),
            torch.randn(8, lexicon.size),
            lexicon,
            lexicon.expected_signs(),
            torch.arange(8) % 4,
            4,
        )
    assert outcome_step is not None
