"""Mini-batch engine.

Ref: Algorithm 1 steps 5-8 -- within one epoch the batch is drawn with modality dropout applied
per observation, the outcome term and the programme penalty are stepped, then a supervised
region batch contributes an alignment step, and finally the invariance penalty contributes its
own step. The three steps are kept separate here rather than summed into one backward pass,
because keeping them separate is what lets the feasibility floor be checked against the
outcome term alone.

Ref: Algorithm 1 step 9-11 -- the feasibility check runs once per epoch, so the engine returns
per-epoch summaries and the veto lives in ``fitting.stages``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from prism.cohort.lexicon import Lexicon
from prism.cohort.schema import Cohort
from prism.cohort.supervision import AlignmentBatch
from prism.cohort.tensors import (
    build_batch,
    labels_for,
    metabolite_for,
    move_batch,
    site_index_for,
)
from prism.errors import ConfigError, SupervisionError
from prism.modelling.assembly import Prism
from prism.modelling.backbone import TensorBatch
from prism.modelling.modality_dropout import draw_dropout_plan
from prism.objectives.invariance import AdversarialPenalty, VariancePenalty
from prism.objectives.total import ConstrainedObjective
from prism.provenance.config import TrainConfig
from prism.provenance.journal import get_logger


@dataclass(frozen=True)
class BatchSpec:
    """Windowing and accumulation."""

    batch_size: int
    grad_accum: int
    shuffle: bool
    seed: int

    def validate(self) -> None:
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ConfigError("batch size and accumulation window must be positive")

    def windows(self, n_rows: int) -> list[np.ndarray]:
        self.validate()
        if n_rows < 0:
            raise ConfigError("a row count cannot be negative")
        return [
            np.arange(start, min(start + self.batch_size, n_rows), dtype=np.int64)
            for start in range(0, n_rows, self.batch_size)
        ]

    def shuffled(self, indices: np.ndarray, epoch: int) -> np.ndarray:
        """Permute row positions for one epoch, deterministically from the run seed."""
        self.validate()
        if not self.shuffle:
            return indices
        rng = np.random.default_rng(self.seed + 7919 * epoch)
        order = rng.permutation(indices.size)
        permuted: np.ndarray = indices[order]
        return permuted

    @classmethod
    def from_config(cls, train: TrainConfig, *, shuffle: bool) -> BatchSpec:
        return cls(
            batch_size=train.batch_size,
            grad_accum=train.grad_accum,
            shuffle=shuffle,
            seed=train.seed,
        )


@dataclass
class EpochSummary:
    """Mean losses and step counts for one epoch."""

    epoch: int
    stage: str
    n_windows: int = 0
    n_rows: int = 0
    term_totals: Mapping[str, float] = field(default_factory=dict)
    optimiser_steps: int = 0
    dropped_modalities: int = 0
    rescued_rows: int = 0
    grad_norms: list[float] = field(default_factory=list)

    def means(self) -> Mapping[str, float]:
        if self.n_windows == 0:
            return dict(self.term_totals)
        return {key: value / float(self.n_windows) for key, value in self.term_totals.items()}

    def total_loss(self) -> float:
        return float(self.means().get("total", 0.0))

    def outcome_loss(self) -> float:
        return float(self.means().get("outcome", 0.0))

    def as_dict(self) -> Mapping[str, float | str]:
        merged: dict[str, float | str] = {
            "epoch": float(self.epoch),
            "stage": self.stage,
            "n_windows": float(self.n_windows),
            "n_rows": float(self.n_rows),
            "optimiser_steps": float(self.optimiser_steps),
            "dropped_modalities": float(self.dropped_modalities),
            "rescued_rows": float(self.rescued_rows),
        }
        for key, value in self.means().items():
            merged[f"mean_{key}"] = value
        if self.grad_norms:
            merged["mean_grad_norm"] = float(np.mean(self.grad_norms))
        return merged


class _Accumulator:
    """Accumulates one sub-step's gradient over an accumulation window."""

    def __init__(
        self,
        optimiser: torch.optim.Optimizer,
        parameters: Sequence[torch.nn.Parameter],
        window: int,
        grad_clip: float,
    ) -> None:
        self.optimiser = optimiser
        self.parameters = parameters
        self.window = window
        self.grad_clip = grad_clip
        self.pending = 0
        self.steps = 0
        self.norms: list[float] = []
        self.optimiser.zero_grad(set_to_none=True)

    def add(self, loss: Tensor, *, last: bool) -> None:
        scaled = loss / float(self.window)
        torch.autograd.backward(scaled)
        self.pending += 1
        if self.pending == self.window or last:
            norm = torch.nn.utils.clip_grad_norm_(list(self.parameters), self.grad_clip)
            self.norms.append(float(norm))
            self.optimiser.step()
            self.optimiser.zero_grad(set_to_none=True)
            self.steps += 1
            self.pending = 0


def _terms(total: Tensor, breakdown: Mapping[str, float]) -> dict[str, float]:
    merged = {key: float(value) for key, value in breakdown.items()}
    merged["total"] = float(total.detach())
    return merged


def apply_dropout(
    batch: TensorBatch, probability: float, generator: torch.Generator
) -> tuple[TensorBatch, int, int]:
    """Apply modality dropout, reporting how many streams were dropped and rows rescued."""
    plan = draw_dropout_plan(batch.present, probability, generator)
    return batch.restrict(plan), plan.n_dropped, plan.n_rescued


def outcome_step(
    model: Prism,
    objective: ConstrainedObjective,
    batch: TensorBatch,
    labels: Tensor,
    lexicon: Lexicon,
    expected_signs: tuple[int, ...],
) -> tuple[Tensor, Mapping[str, float]]:
    """L_Y + lambda_P L_prog, Algorithm 1 step 6."""
    output = model(batch)
    result = objective.stage_one(
        output.outcome_logit, labels, output.programme_coefficients, lexicon, expected_signs
    )
    return result.total, _terms(result.total, result.terms)


def alignment_step(
    model: Prism,
    objective: ConstrainedObjective,
    batch: TensorBatch,
    draw: AlignmentBatch,
    positions: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Mapping[str, float]]:
    """lambda_A L_align on a supervised region batch, Algorithm 1 step 7.

    The metabolite block exists only for the supervised rows of the window, so the decoder output is
    restricted to exactly those rows before the two are matched; matching a restricted metabolite
    block against a whole-window read-out would pair rows from different tissue areas.
    """
    if draw.metabolite.shape[0] != positions.size:
        raise ConfigError("the metabolite block must carry one row per supervised position")
    if int(positions.max(initial=-1)) >= batch.size():
        raise ConfigError("a supervised position falls outside the batch")
    metabolite = torch.from_numpy(np.asarray(draw.metabolite, dtype=np.float32)).to(device)
    read_out = model.read_out(batch)
    selected: Tensor = read_out[torch.from_numpy(positions).to(read_out.device)]
    terms = objective.alignment(selected, metabolite)
    total = objective.weights.lambda_alignment * terms["alignment"]
    return total, _terms(
        total,
        {
            "alignment": float(terms["alignment"].detach()),
            "matching": float(terms["matching"].detach()),
            "second_moment": float(terms["second_moment"].detach()),
        },
    )


def invariance_step(
    model: Prism,
    objective: ConstrainedObjective,
    batch: TensorBatch,
    site_index: Tensor,
    n_sites: int,
    reversal_scale: float,
) -> tuple[Tensor, Mapping[str, float]]:
    """lambda_I L_inv, Algorithm 1 step 8."""
    output = model(batch, reversal_scale=reversal_scale)
    if objective.invariance_mode == "variance":
        terms = VariancePenalty()(output.representation, site_index, n_sites)
    else:
        if output.site_logit is None:
            raise ConfigError("the adversarial reading needs the site classifier output")
        terms = AdversarialPenalty()(output.site_logit, site_index)
    total = objective.weights.lambda_invariance * terms["invariance"]
    return total, _terms(total, {"invariance": float(terms["invariance"].detach())})


def supervised_positions(cohort: Cohort, rows: np.ndarray) -> np.ndarray:
    """Positions inside a window whose record carries metabolite supervision."""
    positions = [offset for offset, row in enumerate(rows) if cohort.records[int(row)].supervised()]
    return np.asarray(positions, dtype=np.int64)


def region_draws_for_window(
    cohort: Cohort,
    rows: np.ndarray,
    lexicon_size: int,
) -> AlignmentBatch | None:
    """Build an alignment draw from the supervised rows of one window."""
    positions = supervised_positions(cohort, rows)
    if positions.size < 3:
        return None
    selected = rows[positions]
    blocks: list[np.ndarray] = []
    regions: list[str] = []
    ids: list[str] = []
    for row in selected:
        record = cohort.records[int(row)]
        if not record.supervised():
            raise SupervisionError(f"record {record.record_id} lost its supervision")
        block = np.asarray(record.supervision(), dtype=np.float32).reshape(-1)
        if block.size != lexicon_size:
            raise SupervisionError(
                f"record {record.record_id} carries {block.size} metabolites, the vocabulary "
                f"holds {lexicon_size}"
            )
        blocks.append(block)
        regions.append(record.region)
        ids.append(record.record_id)
    return AlignmentBatch(
        rows=np.arange(positions.size, dtype=np.int64),
        metabolite=np.stack(blocks, axis=0),
        regions=tuple(regions),
        record_ids=tuple(ids),
    )


def run_epoch(
    model: Prism,
    objective: ConstrainedObjective,
    optimiser: torch.optim.Optimizer | None,
    cohort: Cohort,
    indices: np.ndarray,
    spec: BatchSpec,
    lexicon: Lexicon,
    expected_signs: tuple[int, ...],
    train_config: TrainConfig,
    device: torch.device,
    *,
    stage: str,
    epoch: int,
    generator: torch.Generator,
    with_alignment: bool,
    with_invariance: bool,
) -> EpochSummary:
    """Run one epoch of either stage over a row index set."""
    spec.validate()
    windows = spec.windows(indices.size)
    summary = EpochSummary(epoch=epoch, stage=stage, n_rows=int(indices.size))
    if not windows:
        return summary
    site_index_all = site_index_for(cohort, indices)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    outcome_accumulator: _Accumulator | None = None
    alignment_accumulator: _Accumulator | None = None
    invariance_accumulator: _Accumulator | None = None
    if optimiser is not None:
        outcome_accumulator = _Accumulator(
            optimiser, parameters, spec.grad_accum, train_config.grad_clip
        )
        alignment_accumulator = _Accumulator(
            optimiser, parameters, spec.grad_accum, train_config.grad_clip
        )
        invariance_accumulator = _Accumulator(
            optimiser, parameters, spec.grad_accum, train_config.grad_clip
        )

    for window_position, window in enumerate(windows):
        last = window_position == len(windows) - 1
        rows = indices[window]
        batch = build_batch(cohort, rows)
        if train_config.modality_dropout > 0.0:
            batch, dropped, rescued = apply_dropout(batch, train_config.modality_dropout, generator)
            summary.dropped_modalities += dropped
            summary.rescued_rows += rescued
        batch = move_batch(batch, device)
        labels = labels_for(cohort, rows).to(device)

        total, breakdown = outcome_step(model, objective, batch, labels, lexicon, expected_signs)
        if outcome_accumulator is not None:
            outcome_accumulator.add(total, last=last)
        if optimiser is not None and with_alignment and stage == "stage2":
            positions = supervised_positions(cohort, rows)
            draw = region_draws_for_window(cohort, rows, lexicon.size)
            if draw is not None and alignment_accumulator is not None:
                align_loss, align_terms = alignment_step(
                    model, objective, batch, draw, positions, device
                )
                alignment_accumulator.add(align_loss, last=last)
                breakdown = {**breakdown, **align_terms}
        if optimiser is not None and with_invariance and stage == "stage2":
            local_site = site_index_all[window].to(device)
            inv_loss, inv_terms = invariance_step(
                model, objective, batch, local_site, len(cohort.sites()), 1.0
            )
            if invariance_accumulator is not None:
                invariance_accumulator.add(inv_loss, last=last)
            breakdown = {**breakdown, **inv_terms}
        totals = dict(summary.term_totals)
        for key, value in breakdown.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        summary.term_totals = totals
        summary.n_windows += 1

    for accumulator in (
        outcome_accumulator,
        alignment_accumulator,
        invariance_accumulator,
    ):
        if accumulator is None:
            continue
        summary.optimiser_steps += accumulator.steps
        summary.grad_norms.extend(accumulator.norms)
    return summary


def evaluate_logits(
    model: Prism, cohort: Cohort, indices: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Response logits for a row index set, in the order the indices were given."""
    was_training = model.training
    model.eval()
    collected: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            window = indices[start : start + batch_size]
            batch = move_batch(build_batch(cohort, window), device)
            output = model(batch)
            logits: Tensor = output.outcome_logit.detach().to(torch.float64).cpu()
            collected.append(logits.numpy())
    if was_training:
        model.train()
    if not collected:
        return np.zeros(0, dtype=np.float64)
    combined: np.ndarray = np.concatenate(collected)
    return combined


def evaluate_representations(
    model: Prism, cohort: Cohort, indices: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Fused representations for a row index set."""
    was_training = model.training
    model.eval()
    collected: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            window = indices[start : start + batch_size]
            batch = move_batch(build_batch(cohort, window), device)
            fused: Tensor = model.encode(batch).detach().to(torch.float64).cpu()
            collected.append(fused.numpy())
    if was_training:
        model.train()
    if not collected:
        return np.zeros((0, 0), dtype=np.float64)
    combined: np.ndarray = np.concatenate(collected, axis=0)
    return combined


def evaluate_coefficients(
    model: Prism, cohort: Cohort, indices: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Programme coefficients for a row index set."""
    was_training = model.training
    model.eval()
    collected: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            window = indices[start : start + batch_size]
            batch = move_batch(build_batch(cohort, window), device)
            output = model(batch)
            coefficients: Tensor = output.programme_coefficients.detach().to(torch.float64).cpu()
            collected.append(coefficients.numpy())
    if was_training:
        model.train()
    if not collected:
        return np.zeros((0, 0), dtype=np.float64)
    combined: np.ndarray = np.concatenate(collected, axis=0)
    return combined


def evaluate_probabilities(
    model: Prism, cohort: Cohort, indices: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Response probabilities, the form the calibration and decision readings consume."""
    logits = evaluate_logits(model, cohort, indices, batch_size, device)
    probabilities: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
    return probabilities


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def log_epoch(summary: EpochSummary, run_name: str) -> None:
    logger = get_logger("fitting.loop")
    logger.info(
        "%s %s epoch %d windows=%d rows=%d loss=%.5f",
        run_name,
        summary.stage,
        summary.epoch,
        summary.n_windows,
        summary.n_rows,
        summary.total_loss(),
    )


def supervised_metabolite_matrix(cohort: Cohort, rows: np.ndarray, lexicon_size: int) -> np.ndarray:
    """Metabolite rows of a window, stacked, for a caller that needs them directly."""
    positions = supervised_positions(cohort, rows)
    if positions.size == 0:
        raise SupervisionError("the window carries no supervised record")
    selected = rows[positions]
    block = metabolite_for(cohort, selected).numpy()
    if block.shape[1] != lexicon_size:
        raise SupervisionError(
            f"the window carries {block.shape[1]} metabolites, the vocabulary holds "
            f"{lexicon_size}"
        )
    return block
