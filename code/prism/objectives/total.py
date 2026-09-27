"""The constrained objective.

Ref: Methods Sec. 6.1 Eq. (1) -- the objective couples the outcome term, the
spatial-metabolic alignment term, the cross-setting invariance penalty and the programme
penalty under three weights, subject to the discrimination floor of Eq. (2).
Ref: Algorithm 1 -- Stage 1 fits the heads on the outcome term and the programme penalty
alone, and Stage 2 adds the alignment term and the invariance penalty while pursuing the
same objective subject to the floor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from prism.cohort.lexicon import Lexicon
from prism.errors import ConfigError
from prism.objectives.alignment import AlignmentLoss
from prism.objectives.feasibility import FeasibilityFloor
from prism.objectives.invariance import AdversarialPenalty, VariancePenalty
from prism.objectives.outcome import OutcomeLoss
from prism.objectives.programme import ProgrammePenalty


@dataclass(frozen=True)
class ObjectiveWeights:
    """The three weights of Eq. (1)."""

    lambda_alignment: float = 1.0
    lambda_invariance: float = 0.5
    lambda_programme: float = 0.1
    lambda_sign: float = 0.05

    def validate(self) -> None:
        for name, value in (
            ("lambda_alignment", self.lambda_alignment),
            ("lambda_invariance", self.lambda_invariance),
            ("lambda_programme", self.lambda_programme),
            ("lambda_sign", self.lambda_sign),
        ):
            if value < 0.0:
                raise ConfigError(f"{name} cannot be negative")


@dataclass(frozen=True)
class ObjectiveResult:
    """Total and the per-term breakdown, so an ablation can be read term by term."""

    total: Tensor
    terms: Mapping[str, float]
    active: Mapping[str, bool]

    def scaled(self, factor: float) -> Tensor:
        return self.total * factor


@dataclass(frozen=True)
class ConstrainedObjective:
    """Assembles Eq. (1) from the four penalties."""

    weights: ObjectiveWeights
    invariance_mode: str
    alignment: AlignmentLoss
    programme: ProgrammePenalty
    outcome: OutcomeLoss

    @staticmethod
    def build(
        weights: ObjectiveWeights,
        *,
        invariance_mode: str = "variance",
        temperature: float = 0.1,
        covariance_weight: float = 1.0,
    ) -> ConstrainedObjective:
        """Pair the weights with the penalty modules they scale."""
        weights.validate()
        return ConstrainedObjective(
            weights=weights,
            invariance_mode=invariance_mode,
            alignment=AlignmentLoss(temperature=temperature, covariance_weight=covariance_weight),
            programme=ProgrammePenalty(
                group_weight=weights.lambda_programme, sign_weight=weights.lambda_sign
            ),
            outcome=OutcomeLoss(),
        )

    def _programme_terms(
        self, coefficients: Tensor, lexicon: Lexicon, expected_signs: tuple[int, ...]
    ) -> Mapping[str, Tensor]:
        return self.programme(coefficients, lexicon, expected_signs)

    def stage_one(
        self,
        outcome_logits: Tensor,
        labels: Tensor,
        coefficients: Tensor,
        lexicon: Lexicon,
        expected_signs: tuple[int, ...],
    ) -> ObjectiveResult:
        """L_Y + lambda_P L_prog, the Stage 1 objective of Algorithm 1 step 2."""
        outcome = self.outcome(outcome_logits, labels)
        programme_terms = self._programme_terms(coefficients, lexicon, expected_signs)
        total = outcome + programme_terms["programme"]
        return ObjectiveResult(
            total=total,
            terms={
                "outcome": float(outcome.detach()),
                "programme": float(programme_terms["programme"].detach()),
                "alignment": 0.0,
                "invariance": 0.0,
            },
            active={"outcome": True, "programme": True, "alignment": False, "invariance": False},
        )

    def stage_two(
        self,
        outcome_logits: Tensor,
        labels: Tensor,
        representation: Tensor,
        aligned: Tensor,
        metabolite: Tensor,
        coefficients: Tensor,
        lexicon: Lexicon,
        expected_signs: tuple[int, ...],
        site_index: Tensor,
        n_sites: int,
        site_logits: Tensor | None = None,
    ) -> ObjectiveResult:
        """The full objective of Eq. (1).

        ``aligned`` is the record's read-out in the metabolite space and ``metabolite`` is the
        abundance block of the same tissue area; the first term below matches them, which is why the
        two must share a width. The invariance penalty is read on the representation itself, at the
        width the encoder emits, because the setting is a property of the observation rather than of
        the metabolite space.
        """
        outcome = self.outcome(outcome_logits, labels)
        programme_terms = self._programme_terms(coefficients, lexicon, expected_signs)
        alignment_terms = self.alignment(aligned, metabolite)
        if self.invariance_mode == "variance":
            invariance_terms = VariancePenalty()(representation, site_index, n_sites)
        elif self.invariance_mode == "adversarial":
            if site_logits is None:
                raise ConfigError("the adversarial reading needs the site classifier output")
            invariance_terms = AdversarialPenalty()(site_logits, site_index)
        else:
            raise ConfigError(f"unknown invariance reading {self.invariance_mode!r}")

        total = (
            outcome
            + self.weights.lambda_alignment * alignment_terms["alignment"]
            + self.weights.lambda_invariance * invariance_terms["invariance"]
            + programme_terms["programme"]
        )
        return ObjectiveResult(
            total=total,
            terms={
                "outcome": float(outcome.detach()),
                "alignment": float(alignment_terms["alignment"].detach()),
                "matching": float(alignment_terms["matching"].detach()),
                "second_moment": float(alignment_terms["second_moment"].detach()),
                "invariance": float(invariance_terms["invariance"].detach()),
                "programme": float(programme_terms["programme"].detach()),
                "group": float(programme_terms["group"].detach()),
                "sign": float(programme_terms["sign"].detach()),
            },
            active={
                "outcome": True,
                "alignment": True,
                "invariance": True,
                "programme": True,
            },
        )


def zero_terms() -> Mapping[str, float]:
    """Term names the harness reports even when a weight has been set to zero."""
    return {
        "outcome": 0.0,
        "alignment": 0.0,
        "matching": 0.0,
        "second_moment": 0.0,
        "invariance": 0.0,
        "programme": 0.0,
        "group": 0.0,
        "sign": 0.0,
    }


def blend(mappings: list[Mapping[str, float]]) -> Mapping[str, float]:
    """Sum several term dictionaries over a shared key set."""
    merged = dict(zero_terms())
    for mapping in mappings:
        for key, value in mapping.items():
            merged[key] = merged.get(key, 0.0) + value
    return merged


def effective_floor(raw_floor: float, weights: ObjectiveWeights) -> FeasibilityFloor:
    """Floor and weights travel together: a run reports both or neither.

    Ref: Methods Sec. 6.1 -- the pair (D_min, lambda_I) names a point on the
    invariance-discrimination frontier, so they are validated as a pair rather than
    independently.
    """
    weights.validate()
    if weights.lambda_invariance == 0.0 and raw_floor > 0.0:
        raise ConfigError(
            "an invariance weight of zero makes the floor vacuous; report the arm as "
            "unconstrained instead of constraining nothing"
        )
    return FeasibilityFloor(d_min=raw_floor)
