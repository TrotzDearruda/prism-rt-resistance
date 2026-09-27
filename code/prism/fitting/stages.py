"""Stage orchestration.

Ref: Algorithm 1 -- Stage 1 fits the heads with the encoders frozen on the outcome term and the
programme penalty; Stage 2 releases the encoders and pursues the objective of Eq. (1) subject to
the feasibility constraint of Eq. (2), reverting any epoch whose update would breach the
pre-declared discrimination floor. Early stopping is taken on the development split only, and
the held-out partitions are never inspected inside the fitting loop.

Ref: Methods Sec. 6.9 -- the floor, the equivalence margin and the null band are declared before
any model is fitted, so they are read from the run configuration rather than chosen here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from prism.cohort.lexicon import Lexicon
from prism.cohort.schema import Cohort, Layer
from prism.errors import ConfigError
from prism.fitting.checkpoint import CheckpointMetadata, save_checkpoint
from prism.fitting.loop import (
    BatchSpec,
    EpochSummary,
    evaluate_logits,
    log_epoch,
    make_generator,
    run_epoch,
)
from prism.fitting.optim import OptimiserSettings, build_optimiser
from prism.fitting.schedule import (
    ScheduleSettings,
    attach_scheduler,
    total_steps_for_stages,
)
from prism.measurement.discrimination import rank_auc
from prism.modelling.assembly import Prism
from prism.numerics.seed import SeedState, set_seed
from prism.objectives.feasibility import (
    FeasibilityFloor,
    FeasibilityLedger,
    FeasibilityVerdict,
    evaluate_epoch,
    state_snapshot,
)
from prism.objectives.total import ConstrainedObjective, ObjectiveWeights
from prism.provenance.config import ExperimentConfig
from prism.provenance.journal import get_logger
from prism.provenance.paths import RunPaths


@dataclass(frozen=True)
class DevelopmentSplit:
    """The development arm split into a fitting block and an early-stopping block."""

    fit_rows: np.ndarray
    dev_rows: np.ndarray

    def validate(self) -> None:
        if self.fit_rows.size == 0 or self.dev_rows.size == 0:
            raise ConfigError("the development split needs rows on both sides")


def development_split(cohort: Cohort, dev_fraction: float, seed: int) -> DevelopmentSplit:
    """Split the development arm by patient so no patient straddles the two blocks."""
    patients = sorted(set(cohort.patient_ids()))
    if len(patients) < 2:
        raise ConfigError("the development arm needs at least two patients to be split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    cut = max(1, min(len(patients) - 1, int(round(dev_fraction * len(patients)))))
    dev_patients = {patients[int(index)] for index in order[:cut]}
    fit_rows: list[int] = []
    dev_rows: list[int] = []
    for position, record in enumerate(cohort.records):
        if record.patient_id in dev_patients:
            dev_rows.append(position)
        else:
            fit_rows.append(position)
    split = DevelopmentSplit(
        fit_rows=np.asarray(fit_rows, dtype=np.int64),
        dev_rows=np.asarray(dev_rows, dtype=np.int64),
    )
    split.validate()
    return split


@dataclass
class StageTrace:
    """Per-epoch record of one stage."""

    stage: str
    summaries: list[EpochSummary] = field(default_factory=list)
    dev_discrimination: list[float] = field(default_factory=list)
    dev_loss: list[float] = field(default_factory=list)
    verdicts: list[FeasibilityVerdict] = field(default_factory=list)
    best_epoch: int = -1
    best_value: float = float("-inf")
    stopped_early: bool = False

    def record(self, summary: EpochSummary, dev_auc: float, dev_loss: float) -> None:
        self.summaries.append(summary)
        self.dev_discrimination.append(dev_auc)
        self.dev_loss.append(dev_loss)

    def note_best(self, epoch: int, value: float) -> None:
        if value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch

    def best_summary(self) -> EpochSummary | None:
        for summary in self.summaries:
            if summary.epoch == self.best_epoch:
                return summary
        return None

    def as_dict(self) -> Mapping[str, object]:
        return {
            "stage": self.stage,
            "epochs": len(self.summaries),
            "best_epoch": self.best_epoch,
            "best_development_discrimination": self.best_value,
            "stopped_early": self.stopped_early,
            "vetoes": sum(1 for verdict in self.verdicts if not verdict.admitted),
            "development_discrimination": list(self.dev_discrimination),
            "development_loss": list(self.dev_loss),
        }


@dataclass
class FitResult:
    """Everything a run needs to report what the fitting did."""

    model: Prism
    stage1: StageTrace
    stage2: StageTrace
    ledger: FeasibilityLedger
    checkpoints: list[CheckpointMetadata]
    seed_state: SeedState

    def as_dict(self) -> Mapping[str, object]:
        return {
            "seed": self.seed_state.seed,
            "deterministic": self.seed_state.deterministic,
            "stage1": self.stage1.as_dict(),
            "stage2": self.stage2.as_dict(),
            "feasibility": dict(self.ledger.summary()),
            "checkpoints": [metadata.as_dict() for metadata in self.checkpoints],
        }


def _optimiser_settings(config: ExperimentConfig) -> OptimiserSettings:
    return OptimiserSettings(
        lr_head=config.train.lr_head,
        lr_encoder=config.train.lr_encoder,
        weight_decay=config.train.weight_decay,
        grad_clip=config.train.grad_clip,
    )


def _dev_loss(
    model: Prism, cohort: Cohort, rows: np.ndarray, batch_size: int, device: torch.device
) -> float:
    logits = evaluate_logits(model, cohort, rows, batch_size, device)
    labels = np.asarray([cohort.records[int(row)].outcome for row in rows], dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _dev_discrimination(
    model: Prism, cohort: Cohort, rows: np.ndarray, batch_size: int, device: torch.device
) -> float:
    logits = evaluate_logits(model, cohort, rows, batch_size, device)
    labels = np.asarray([cohort.records[int(row)].outcome for row in rows], dtype=np.int64)
    if np.unique(labels).size < 2:
        return 0.5
    return rank_auc(logits, labels)


def _checkpoint(
    paths: RunPaths,
    model: Prism,
    config: ExperimentConfig,
    seed_state: SeedState,
    *,
    member: int,
    stage: str,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    best: bool = False,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> CheckpointMetadata:
    metadata = {
        "run": config.name,
        "member": member,
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "extra": {"arm": config.arm, "invariance_mode": config.objective.invariance_mode},
    }
    return save_checkpoint(
        paths.checkpoint_file(member, best=best),
        model,
        seed_state,
        metadata,
        optimiser=optimiser,
        scheduler=scheduler,
    )


def fit(
    config: ExperimentConfig,
    parts: Mapping[Layer, Cohort],
    lexicon: Lexicon,
    device: torch.device,
    paths: RunPaths,
    *,
    member: int = 0,
    dev_fraction: float = 0.15,
    early_stopping_tolerance: float = 1e-4,
) -> FitResult:
    """Fit one member of the ensemble through the two stages of Algorithm 1."""
    if Layer.DEVELOPMENT not in parts:
        raise ConfigError("the development arm is required to fit anything")
    seed_state = set_seed(config.train.seed + 101 * member)
    objective = ConstrainedObjective.build(
        ObjectiveWeights(
            lambda_alignment=config.objective.lambda_alignment,
            lambda_invariance=config.objective.lambda_invariance,
            lambda_programme=config.objective.lambda_programme,
            lambda_sign=config.objective.lambda_sign,
        ),
        invariance_mode=config.objective.invariance_mode,
        temperature=config.objective.matching_temperature,
        covariance_weight=config.objective.covariance_weight,
    )
    model = Prism(config.model, lexicon, arm=config.arm).to(device)
    expected_signs = lexicon.expected_signs()
    development = parts[Layer.DEVELOPMENT]
    split = development_split(development, dev_fraction, config.train.seed + member)
    spec = BatchSpec.from_config(config.train, shuffle=True)
    generator = make_generator(config.train.seed + 17 * member, device)
    steps_stage1, steps_stage2 = total_steps_for_stages(
        split.fit_rows.size,
        config.train.batch_size,
        config.train.grad_accum,
        config.train.stage1_epochs,
        config.train.stage2_epochs,
    )
    settings = _optimiser_settings(config)
    trace1 = StageTrace(stage="stage1")
    trace2 = StageTrace(stage="stage2")
    ledger = FeasibilityLedger(floor=FeasibilityFloor(d_min=config.train.discrimination_floor))
    checkpoints: list[CheckpointMetadata] = []
    logger = get_logger("fitting.stages")

    model.freeze_encoders()
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        settings,
        include_encoder=False,
    )
    scheduler = attach_scheduler(
        optimiser,
        ScheduleSettings(total_steps=steps_stage1, warmup_fraction=config.train.warmup_fraction),
    )
    for epoch in range(config.train.stage1_epochs):
        summary = run_epoch(
            model,
            objective,
            optimiser,
            development,
            split.fit_rows,
            spec,
            lexicon,
            expected_signs,
            config.train,
            device,
            stage="stage1",
            epoch=epoch,
            generator=generator,
            with_alignment=False,
            with_invariance=False,
        )
        scheduler.step()
        log_epoch(summary, config.name)
        dev_auc = _dev_discrimination(
            model, development, split.dev_rows, config.train.batch_size, device
        )
        dev_loss = _dev_loss(model, development, split.dev_rows, config.train.batch_size, device)
        trace1.record(summary, dev_auc, dev_loss)
        trace1.note_best(epoch, dev_auc)
        checkpoints.append(
            _checkpoint(
                paths,
                model,
                config,
                seed_state,
                member=member,
                stage="stage1",
                epoch=epoch,
                global_step=(epoch + 1) * summary.optimiser_steps,
                metrics={"loss": summary.total_loss(), "dev_auc": dev_auc},
            )
        )
    logger.info("stage 1 finished at dev discrimination %.4f", trace1.best_value)

    model.unfreeze_encoders()
    optimiser = build_optimiser(
        model.head_parameters(),
        list(model.backbone.parameters()),
        settings,
        include_encoder=True,
    )
    scheduler = attach_scheduler(
        optimiser,
        ScheduleSettings(total_steps=steps_stage2, warmup_fraction=config.train.warmup_fraction),
    )
    stale = 0
    for epoch in range(config.train.stage2_epochs):
        snapshot = state_snapshot(model)
        summary = run_epoch(
            model,
            objective,
            optimiser,
            development,
            split.fit_rows,
            spec,
            lexicon,
            expected_signs,
            config.train,
            device,
            stage="stage2",
            epoch=epoch,
            generator=generator,
            with_alignment=True,
            with_invariance=config.arm != "unconstrained",
        )
        scheduler.step()
        log_epoch(summary, config.name)
        dev_auc = _dev_discrimination(
            model, development, split.dev_rows, config.train.batch_size, device
        )
        verdict = evaluate_epoch(model, epoch, dev_auc, ledger.floor, snapshot)
        trace2.verdicts.append(verdict)
        ledger.verdicts.append(verdict)
        if not verdict.admitted:
            dev_auc = _dev_discrimination(
                model, development, split.dev_rows, config.train.batch_size, device
            )
        dev_loss = _dev_loss(model, development, split.dev_rows, config.train.batch_size, device)
        trace2.record(summary, dev_auc, dev_loss)
        previous_best = trace2.best_value
        trace2.note_best(epoch, dev_auc)
        if trace2.best_value > previous_best + early_stopping_tolerance:
            stale = 0
        else:
            stale += 1
        checkpoints.append(
            _checkpoint(
                paths,
                model,
                config,
                seed_state,
                member=member,
                stage="stage2",
                epoch=epoch,
                global_step=steps_stage1 + (epoch + 1) * summary.optimiser_steps,
                metrics={
                    "loss": summary.total_loss(),
                    "dev_auc": dev_auc,
                    "admitted": float(verdict.admitted),
                },
                optimiser=optimiser,
                scheduler=scheduler,
            )
        )
        if trace2.best_epoch == epoch:
            _checkpoint(
                paths,
                model,
                config,
                seed_state,
                member=member,
                stage="stage2-best",
                epoch=epoch,
                global_step=steps_stage1 + (epoch + 1) * summary.optimiser_steps,
                metrics={"loss": summary.total_loss(), "dev_auc": dev_auc},
                best=True,
            )
        if stale >= config.train.patience:
            trace2.stopped_early = True
            logger.info("stage 2 stopped early at epoch %d", epoch)
            break

    return FitResult(
        model=model,
        stage1=trace1,
        stage2=trace2,
        ledger=ledger,
        checkpoints=checkpoints,
        seed_state=seed_state,
    )


def select_best_epoch(trace: StageTrace) -> int:
    if trace.best_epoch < 0:
        raise ConfigError("no epoch was recorded as best")
    return trace.best_epoch


def convergence_reading(traces: Sequence[StageTrace]) -> Mapping[str, float]:
    """End-of-run reading of where each stage settled."""
    reading: dict[str, float] = {}
    for trace in traces:
        if not trace.dev_discrimination:
            continue
        reading[f"{trace.stage}_final_dev_discrimination"] = trace.dev_discrimination[-1]
        reading[f"{trace.stage}_best_dev_discrimination"] = trace.best_value
        reading[f"{trace.stage}_epochs"] = float(len(trace.summaries))
    return reading
