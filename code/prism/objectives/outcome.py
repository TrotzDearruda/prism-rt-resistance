"""Outcome term.

Ref: Methods Sec. 6.1 Eq. (1), first summand -- L_Y(Y, g(Z)) is the deterioration caused
by the response, so the head is trained with a logistic loss on the response and nothing
else. The loss is expressed in logits rather than probabilities because the outcome head
emits a logit and a probability transform in the loss would lose the numerical range the
Stage 2 feasibility check relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from prism.errors import ConfigError


@dataclass(frozen=True)
class OutcomeLoss:
    """Logistic loss with the per-class weighting made explicit."""

    positive_weight: float = 1.0

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.dim() != 1 or labels.dim() != 1:
            raise ConfigError("the outcome term reads one logit and one label per row")
        if logits.shape[0] != labels.shape[0]:
            raise ConfigError("the outcome term needs aligned logits and labels")
        if labels.numel() == 0:
            raise ConfigError("the outcome term needs at least one label")
        weight: Tensor | None = None
        if self.positive_weight != 1.0:
            weight = torch.where(
                labels > 0.5,
                torch.full_like(logits, self.positive_weight),
                torch.ones_like(logits),
            )
        loss: Tensor = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), weight=weight, reduction="mean"
        )
        return loss

    def from_probability(self, probabilities: Tensor, labels: Tensor) -> Tensor:
        """Same loss for a caller that already holds probabilities."""
        clipped = probabilities.clamp(1e-6, 1.0 - 1e-6)
        logits: Tensor = torch.log(clipped) - torch.log1p(-clipped)
        return self(logits, labels)


def implied_prevalence(probabilities: Tensor) -> float:
    """Mean predicted probability, read against the reported response rate."""
    if probabilities.numel() == 0:
        raise ConfigError("a prevalence needs at least one probability")
    mean: Tensor = probabilities.detach().mean()
    return float(mean)


def balanced_positive_weight(labels: Tensor) -> float:
    """Weight that equalises the two classes, for the arms that need it."""
    positives = float((labels > 0.5).sum())
    negatives = float(labels.numel() - positives)
    if positives == 0.0:
        raise ConfigError("the outcome term cannot be weighted with no positive label")
    return negatives / positives


def brier_decomposition_free(probabilities: Tensor, labels: Tensor) -> float:
    """Mean squared error of the response probabilities.

    Ref: Table 1 -- a Brier score is reported beside the calibration slope, and it is
    computed on probabilities rather than on logits.
    """
    if probabilities.shape != labels.shape:
        raise ConfigError("the Brier score needs aligned probabilities and labels")
    delta: Tensor = probabilities - labels.to(probabilities.dtype)
    squared: Tensor = delta * delta
    return float(squared.mean())


def outcome_accuracy(logits: Tensor, labels: Tensor, threshold: float = 0.0) -> float:
    predicted: Tensor = (logits > threshold).to(labels.dtype)
    agreement: Tensor = (predicted == labels).to(torch.float32)
    return float(agreement.mean())
