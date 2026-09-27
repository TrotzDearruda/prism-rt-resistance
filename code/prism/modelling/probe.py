"""Outcome head and the linear probe baseline.

Ref: Algorithm 1 Stage 1 -- the encoders are frozen and the heads are fitted on the
outcome term and the programme penalty. Ref: Table 1 -- a frozen representation read by a
linear probe is a reported baseline in its own right, so the probe is a first-class module
rather than a debugging aid. Ref: Table 6 -- the probe column isolates the transferable
content of the frozen representation from what the constrained objective contributes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.errors import ConfigError


class OutcomeHead(nn.Module):
    """g: fused representation to a response logit."""

    net: nn.Sequential

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim < 1 or hidden < 1:
            raise ConfigError("an outcome head needs positive input and hidden widths")
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, representation: Tensor) -> Tensor:
        logit: Tensor = self.net(representation).squeeze(-1)
        return logit


class LinearProbe(nn.Module):
    """A single affine read-out over a frozen representation."""

    linear: nn.Linear

    def __init__(self, in_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        if in_dim < 1:
            raise ConfigError("a probe needs a positive input width")
        self.linear = nn.Linear(in_dim, 1, bias=bias)

    def forward(self, representation: Tensor) -> Tensor:
        logit: Tensor = self.linear(representation).squeeze(-1)
        return logit


@dataclass(frozen=True)
class ProbeFit:
    """Coefficients of a closed-form probe, kept next to its training residual."""

    coefficients: Tensor
    intercept: float
    ridge: float
    train_accuracy: float
    n_rows: int


def _standardise(features: Tensor) -> Tensor:
    centred: Tensor = features - features.mean(dim=0, keepdim=True)
    scale: Tensor = centred.std(dim=0, keepdim=True).clamp(min=1e-6)
    return centred / scale


def fit_probe_closed_form(
    features: Tensor,
    labels: Tensor,
    *,
    ridge: float = 1e-3,
) -> ProbeFit:
    """Ridge least squares on standardised features, with an explicit intercept.

    The probe is fitted without gradient steps so that a reported probe number cannot
    depend on an optimisation schedule; standardisation happens inside the fit so that
    coefficients are comparable across representations with different scales.
    """
    if features.dim() != 2 or labels.dim() != 1:
        raise ConfigError("a probe fit expects (rows, features) and (rows,)")
    if features.shape[0] != labels.shape[0]:
        raise ConfigError("probe features and labels must share a row count")
    n_rows, n_features = features.shape
    if n_rows <= n_features:
        ridge = max(ridge, 1e-2)
    standardised = _standardise(features)
    design: Tensor = torch.cat(
        [torch.ones((n_rows, 1), dtype=features.dtype, device=features.device), standardised],
        dim=1,
    )
    targets: Tensor = labels.to(features.dtype)
    penalty: Tensor = ridge * torch.eye(
        n_features + 1, dtype=features.dtype, device=features.device
    )
    penalty[0, 0] = 0.0
    gram: Tensor = design.T @ design + penalty
    solution: Tensor = torch.linalg.solve(gram, design.T @ targets)
    scores: Tensor = design @ solution
    predicted: Tensor = (scores > 0.5).to(labels.dtype)
    agreement: Tensor = (predicted == labels).to(torch.float32)
    accuracy = float(agreement.mean())
    return ProbeFit(
        coefficients=solution[1:],
        intercept=float(solution[0]),
        ridge=ridge,
        train_accuracy=accuracy,
        n_rows=int(n_rows),
    )


def probe_scores(features: Tensor, fit: ProbeFit) -> Tensor:
    """Apply a closed-form probe to a representation matrix."""
    standardised = _standardise(features)
    scores: Tensor = standardised @ fit.coefficients + fit.intercept
    return scores


def probe_from_state(fit: ProbeFit, bias: bool = True) -> LinearProbe:
    """Rebuild a module around fitted coefficients so a probe can be evaluated with the
    same code path as a trained head."""
    probe = LinearProbe(int(fit.coefficients.shape[0]), bias=bias)
    with torch.no_grad():
        probe.linear.weight.copy_(fit.coefficients.reshape(1, -1))
        if bias and probe.linear.bias is not None:
            probe.linear.bias.fill_(fit.intercept)
    return probe


def head_gradient_norm(head: nn.Module) -> float:
    """L2 norm of the head's gradients, reported so a stalled Stage 1 is visible."""
    total = 0.0
    for parameter in head.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        squared: Tensor = gradient.detach().pow(2)
        total += float(squared.sum())
    return math.sqrt(total)
