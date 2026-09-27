"""Reader study.

Ref: Table 4 -- three arms are read on the same cases: a clinician alone, the model alone and
a clinician with the model output in front of them. Ref: Results Sec. 2.3 -- the three-arm
form exists because the pairwise clinician-versus-model contrast cannot say whether the
combination helps, and the overcall rate is the quantity watched for automation bias, which
the study expects to fall rather than rise once the model output is available. Ref: Results
Sec. 2.3 again -- the clinician-alone arm is a three-reader arm, which is why inter-reader
agreement is reported next to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.errors import ConfigError
from prism.measurement.discrimination import rank_auc
from prism.numerics.operating_point import (
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)

ARMS: tuple[str, ...] = ("clinician-alone", "model-alone", "clinician-with-model")


@dataclass(frozen=True)
class ReaderCase:
    """One case with the calls it received and the reference reading."""

    case_id: str
    reference: int
    clinician_calls: tuple[int, ...]
    model_call: int


def overcall_rate(calls: np.ndarray, references: np.ndarray) -> float:
    """Share of cases called positive where the reference is not.

    The quantity is read as the false-positive share of the calls, which is the direction a
    reader drifts when a model output is in front of them.
    """
    values = np.asarray(calls).reshape(-1)
    truth = np.asarray(references).reshape(-1)
    if values.size != truth.size or values.size == 0:
        raise ConfigError("an overcall rate needs aligned calls and references")
    negative = truth <= 0
    if not negative.any():
        raise ConfigError("an overcall rate needs at least one negative reference case")
    return float((values[negative] > 0).mean())


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Agreement across raters on categorical calls.

    ``ratings`` is (n_items, n_categories) with counts of raters per category. The statistic
    is reported for the clinician-alone arm because that arm answers a different question from
    the other two.
    """
    table = np.asarray(ratings, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] == 0:
        raise ConfigError("a kappa table is items by categories")
    per_item = table.sum(axis=1)
    if np.any(per_item < 2):
        raise ConfigError("a kappa needs at least two raters on every item")
    n_raters = float(per_item.mean())
    proportions = table / per_item[:, None]
    agreement = (np.sum(proportions * proportions, axis=1) - 1.0 / n_raters) / (
        1.0 - 1.0 / n_raters
    )
    observed = float(agreement.mean())
    marginal = table.sum(axis=0) / table.sum()
    expected = float(np.sum(marginal * marginal))
    if abs(1.0 - expected) < 1e-12:
        raise ConfigError("a kappa is undefined when one category absorbs every rating")
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class ReaderArmResult:
    """One arm of the reader table."""

    arm: str
    auc: float
    sensitivity: float
    specificity: float
    overcall: float
    half_width: float

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "arm": self.arm,
            "auc": self.auc,
            "sensitivity_at_fixed_specificity": self.sensitivity,
            "specificity_at_fixed_sensitivity": self.specificity,
            "overcall_rate": self.overcall,
            "pi_half_width": self.half_width,
        }


def _reader_scores(cases: Sequence[ReaderCase], arm: str) -> np.ndarray:
    if arm == "model-alone":
        return np.asarray([case.model_call for case in cases], dtype=np.float64)
    if arm == "clinician-with-model":
        return np.asarray(
            [0.5 * (case.model_call + float(np.mean(case.clinician_calls))) for case in cases],
            dtype=np.float64,
        )
    if arm == "clinician-alone":
        return np.asarray(
            [float(np.mean(case.clinician_calls)) for case in cases], dtype=np.float64
        )
    raise ConfigError(f"unknown reader arm {arm!r}")


def reader_arm_result(
    cases: Sequence[ReaderCase], arm: str, half_width: float, target_specificity: float = 0.8
) -> ReaderArmResult:
    if arm not in ARMS:
        raise ConfigError(f"the reader study has no arm named {arm!r}")
    references = np.asarray([case.reference for case in cases], dtype=np.int64)
    scores = _reader_scores(cases, arm)
    calls = (scores > 0.5).astype(np.int64)
    sensitivity = sensitivity_at_specificity(scores, references, target_specificity)
    specificity = specificity_at_sensitivity(scores, references, sensitivity)
    return ReaderArmResult(
        arm=arm,
        auc=rank_auc(scores, references),
        sensitivity=sensitivity,
        specificity=specificity,
        overcall=overcall_rate(calls, references),
        half_width=half_width,
    )


def inter_reader_table(cases: Sequence[ReaderCase], n_categories: int = 2) -> float:
    """Fleiss kappa over the clinician calls in the clinician-alone arm."""
    rows = np.zeros((len(cases), n_categories), dtype=np.float64)
    for position, case in enumerate(cases):
        for call in case.clinician_calls:
            category = min(max(int(call), 0), n_categories - 1)
            rows[position, category] += 1.0
    return fleiss_kappa(rows)


def team_benefit(
    arms: Mapping[str, ReaderArmResult], tolerance: float = 0.0
) -> Mapping[str, float]:
    """Whether the combination beats both of its parts.

    Ref: Results Sec. 2.3 -- the claim is about teamwork rather than about the model, so the
    margin is taken against the better of the two single-component arms on every column that
    carries the claim. A tolerance can be applied, and the margin is reported either way, so a
    combination that only ties is visible rather than hidden.
    """
    for arm in ARMS:
        if arm not in arms:
            raise ConfigError(f"the reader table is missing the {arm!r} arm")
    team = arms["clinician-with-model"]
    clinician = arms["clinician-alone"]
    model = arms["model-alone"]
    best_auc = max(clinician.auc, model.auc)
    best_sensitivity = max(clinician.sensitivity, model.sensitivity)
    best_specificity = max(clinician.specificity, model.specificity)
    return {
        "auc_margin": team.auc - best_auc,
        "sensitivity_margin": team.sensitivity - best_sensitivity,
        "specificity_margin": team.specificity - best_specificity,
        "overcall_reduction": model.overcall - team.overcall,
        "beats_both": float(
            team.auc >= best_auc - tolerance
            and team.sensitivity >= best_sensitivity - tolerance
            and team.specificity >= best_specificity - tolerance
        ),
    }


def automation_bias_reading(arms: Mapping[str, ReaderArmResult]) -> Mapping[str, float]:
    """Direction in which the overcall rate moved when the model output appeared.

    Ref: Results Sec. 2.3 -- the study reports the overcall rate as falling rather than rising,
    which is the opposite of the automation-bias direction, so the sign is part of the reading
    rather than a detail of the table.
    """
    for arm in ("clinician-alone", "clinician-with-model"):
        if arm not in arms:
            raise ConfigError(f"the automation reading needs the {arm!r} arm")
    change = arms["clinician-with-model"].overcall - arms["clinician-alone"].overcall
    return {
        "overcall_change": change,
        "falls_with_model_output": float(change < 0.0),
    }
