"""Assembly of the constrained multimodal model.

Ref: Methods Sec. 6.1 -- the learned representation is Z = f(X), the outcome head g maps it
to a response and the programme decoder h maps it to a signed metabolite programme, with a
separate head for the interval around each programme effect. Ref: Methods Sec. 6.5 -- the
adversarial reading of the invariance penalty needs a site classifier trained against the
encoder through a gradient reversal layer, so the classifier exists only in that arm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.cohort.lexicon import Lexicon
from prism.errors import ConfigError
from prism.modelling.backbone import PrismBackbone, TensorBatch
from prism.modelling.interval_head import IntervalHead
from prism.modelling.probe import OutcomeHead
from prism.modelling.programme_head import ProgrammeDecoder
from prism.provenance.config import ModelConfig


def reverse_gradient(inputs: Tensor, scale: float = 1.0) -> Tensor:
    """Identity in the forward direction, negated gradient in the backward direction.

    Ref: Methods Sec. 6.5 -- the site classifier is learned while acting against the
    encoder. Writing the reversal as ``detached + (inputs - detached) * -scale`` keeps the
    forward value equal to ``inputs`` while sending ``-scale`` times the incoming gradient
    back to the encoder, without a custom autograd function.
    """
    detached = inputs.detach()
    reversed_output: Tensor = detached + (inputs - detached) * (-scale)
    return reversed_output


class SiteClassifier(nn.Module):
    """Predicts the care setting from the representation, used by the adversarial arm."""

    net: nn.Sequential
    out: nn.Linear

    def __init__(self, in_dim: int, n_sites: int, hidden: int) -> None:
        super().__init__()
        if n_sites < 2:
            raise ConfigError("a site classifier needs at least two settings to separate")
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.out = nn.Linear(hidden, n_sites)
        self.n_sites = n_sites

    def forward(self, representation: Tensor, scale: float = 1.0) -> Tensor:
        reversed_representation = reverse_gradient(representation, scale)
        hidden: Tensor = self.net(reversed_representation)
        logits: Tensor = self.out(hidden)
        return logits


@dataclass
class PrismOutput:
    """Everything one forward pass produces."""

    representation: Tensor
    outcome_logit: Tensor
    programme_coefficients: Tensor
    programme_beta: Tensor
    programme_scale: Tensor
    site_logit: Tensor | None
    record_ids: tuple[str, ...]


class Prism(nn.Module):
    """The constrained multimodal model."""

    backbone: PrismBackbone
    outcome: OutcomeHead
    decoder: ProgrammeDecoder | None
    interval: IntervalHead | None
    site_classifier: SiteClassifier | None

    def __init__(
        self,
        model: ModelConfig,
        lexicon: Lexicon,
        *,
        arm: str = "constrained",
        n_sites: int = 7,
        presence_gate: bool = True,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.lexicon = lexicon
        self.backbone = PrismBackbone(model.streams, model.fused_dim, presence_gate=presence_gate)
        in_dim = model.fused_dim
        self.outcome = OutcomeHead(in_dim, model.probe_hidden)
        self.decoder = (
            None
            if arm == "backbone-only"
            else ProgrammeDecoder(in_dim, lexicon.size, model.decoder_hidden)
        )
        self.interval = (
            None
            if arm in {"backbone-only", "no-interval"}
            else IntervalHead(in_dim, lexicon.n_blocks(), model.interval_hidden)
        )
        self.site_classifier = (
            SiteClassifier(in_dim, n_sites, model.site_classifier_hidden)
            if arm == "adversarial"
            else None
        )

    def encode(self, batch: TensorBatch) -> Tensor:
        fused: Tensor = self.backbone(batch)
        return fused

    def read_out(self, batch: TensorBatch) -> Tensor:
        """The record's read-out in the metabolite space, which the alignment term matches.

        When the arm carries no decoder there is no metabolite-space read-out, and the encoder's
        own representation is returned instead; the alignment step is skipped for the arms that do
        not carry a decoder, so no reported alignment term takes that path.
        """
        fused = self.encode(batch)
        if self.decoder is None:
            return fused
        coefficients: Tensor = self.decoder(fused)
        return coefficients

    def forward(self, batch: TensorBatch, *, reversal_scale: float = 1.0) -> PrismOutput:
        representation = self.encode(batch)
        outcome_logit: Tensor = self.outcome(representation)
        if self.decoder is None:
            coefficients = torch.zeros(
                (int(representation.shape[0]), self.lexicon.size),
                dtype=representation.dtype,
                device=representation.device,
            )
        else:
            coefficients = self.decoder(representation)
        if self.interval is None:
            beta = torch.zeros(
                (int(representation.shape[0]), self.lexicon.n_blocks()),
                dtype=representation.dtype,
                device=representation.device,
            )
            scale = torch.ones_like(beta)
        else:
            beta, scale = self.interval(representation)
        site_logit: Tensor | None = None
        if self.site_classifier is not None:
            site_logit = self.site_classifier(representation, reversal_scale)
        return PrismOutput(
            representation=representation,
            outcome_logit=outcome_logit,
            programme_coefficients=coefficients,
            programme_beta=beta,
            programme_scale=scale,
            site_logit=site_logit,
            record_ids=batch.record_ids,
        )

    def freeze_encoders(self) -> None:
        """Stage 1 of Algorithm 1 fits the heads with the encoders held fixed."""
        self.backbone.freeze()

    def unfreeze_encoders(self) -> None:
        self.backbone.unfreeze()

    def head_parameters(self) -> list[nn.Parameter]:
        modules: list[nn.Module] = [self.outcome]
        for optional in (self.decoder, self.interval, self.site_classifier):
            if optional is not None:
                modules.append(optional)
        return [parameter for module in modules for parameter in module.parameters()]

    def objective_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def component_names(self) -> Mapping[str, bool]:
        """Which parts of the objective this arm can express."""
        return {
            "alignment": self.decoder is not None,
            "invariance": self.arm != "unconstrained",
            "programme": self.decoder is not None,
            "interval": self.interval is not None,
            "adversarial": self.site_classifier is not None,
        }

    def set_encoder_grad(self, enabled: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(enabled)
