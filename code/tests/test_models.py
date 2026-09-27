"""Encoders, pooling, fusion, heads and the assembled model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.errors import ConfigError
from prism.modelling.assembly import Prism, SiteClassifier, reverse_gradient
from prism.modelling.interval_head import (
    IntervalHead,
    coverage_shift,
    interval_from_scale,
    report_intervals,
    z_multiplier,
)
from prism.modelling.pooling import (
    GatedPool,
    MaskedAttentionPool,
    MaxPool,
    MeanPool,
    build_pooling,
)
from prism.modelling.probe import (
    LinearProbe,
    OutcomeHead,
    fit_probe_closed_form,
    probe_scores,
)
from prism.modelling.programme_head import (
    ProgrammeDecoder,
    SupportThreshold,
    emissions_from_coefficients,
    select_support,
    sign_agreement,
    signs_for_support,
    support_sizes,
)
from prism.modelling.streams import build_stream, stream_output_width
from prism.notation import MODALITY_STREAMS
from prism.provenance.config import ModelConfig, StreamConfig


def test_each_stream_emits_one_row_per_observation() -> None:
    cases = {
        "ct": ([1, 2, 4, 4], (3, 1, 2, 4, 4)),
        "mri": ([1, 2, 4, 4], (3, 1, 2, 4, 4)),
        "histology": ([16], (3, 16)),
        "labs": ([6], (3, 6)),
    }
    for modality, (shape, inputs) in cases.items():
        stream = build_stream(modality, tuple(shape), 8, 2)
        assert stream_output_width(stream) == 8
        output = stream(torch.randn(*inputs))
        assert output.shape == (3, 8)


def test_a_volume_stream_refuses_a_block_mask() -> None:
    stream = build_stream("ct", (1, 2, 4, 4), 8, 1)
    with pytest.raises(ConfigError):
        stream(torch.randn(2, 1, 2, 4, 4), torch.ones(2, 3, dtype=torch.bool))


def test_pooling_variants_agree_on_shape() -> None:
    blocks = torch.randn(5, 7, 4)
    mask = torch.ones(5, 7, dtype=torch.bool)
    mask[0, 3:] = False
    for kind in ("attention", "mean", "max", "gated"):
        pooled = build_pooling(kind, 4)(blocks, mask)
        assert pooled.shape == (5, 4)
    assert MaskedAttentionPool(4)(blocks, mask).shape == (5, 4)
    assert MeanPool()(blocks).shape == (5, 4)
    assert MaxPool()(blocks).shape == (5, 4)
    assert GatedPool(4)(blocks, mask).shape == (5, 4)


def test_masked_pooling_ignores_padded_blocks() -> None:
    blocks = torch.zeros(1, 4, 3)
    blocks[0, :2] = 1.0
    blocks[0, 2:] = 9.0
    mask = torch.tensor([[True, True, False, False]])
    averaged = MeanPool()(blocks, mask)
    assert torch.allclose(averaged, torch.ones(1, 3), atol=1e-6)
    attended = MaskedAttentionPool(3)(blocks, mask)
    assert attended.shape == (1, 3)


def test_probe_recovers_a_linear_signal_and_is_at_chance_without_one() -> None:
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(200, 12, generator=generator)
    weights = torch.randn(12, generator=generator)
    labels = (features @ weights > 0).to(torch.int64)
    fit_result = fit_probe_closed_form(features, labels, ridge=1e-3)
    scores = probe_scores(features, fit_result)
    accuracy = float(((scores > 0.5).to(labels.dtype) == labels).to(torch.float32).mean())
    assert accuracy > 0.9
    scrambled = torch.randperm(200, generator=generator)
    chance = float(((scores > 0.5).to(labels.dtype) == labels[scrambled]).to(torch.float32).mean())
    assert chance < 0.8


def test_outcome_head_and_probe_emit_one_logit_per_row() -> None:
    representation = torch.randn(6, 10)
    assert OutcomeHead(10, 8)(representation).shape == (6,)
    assert LinearProbe(10)(representation).shape == (6,)


def test_programme_support_respects_its_bounds(lexicon) -> None:
    coefficients = torch.randn(9, lexicon.size) * 0.4
    threshold = SupportThreshold(magnitude=0.15, per_block_min=1, per_block_max=4)
    support = select_support(coefficients, lexicon, threshold)
    sizes = support_sizes(support, lexicon)
    assert len(support) == 9
    assert lexicon.n_blocks() <= sizes["overall"] <= 4 * lexicon.n_blocks()
    for indices in support:
        assert indices == tuple(sorted(indices))
    signs = signs_for_support(coefficients, support)
    assert all(sign in {-1, 1} for row in signs for sign in row)


def test_the_decoder_emits_a_signed_programme(lexicon) -> None:
    decoder = ProgrammeDecoder(12, lexicon.size, 8)
    coefficients = decoder(torch.randn(4, 12))
    assert coefficients.shape == (4, lexicon.size)
    statements = emissions_from_coefficients(
        coefficients, lexicon, tuple(f"R{index}" for index in range(4))
    )
    assert len(statements) == 4
    for statement in statements:
        assert statement.supported
        assert set(statement.signs) == set(statement.supported)
        assert statement.programme_support


def test_sign_agreement_uses_the_reported_directions(lexicon) -> None:
    expected = {entry.name: entry.expected_sign for entry in lexicon.catalogue()}
    agreeing = torch.zeros(3, lexicon.size)
    for entry in lexicon.catalogue():
        agreeing[:, entry.index] = float(entry.expected_sign) * 0.5
    assert sign_agreement(agreeing, lexicon, expected) == pytest.approx(1.0)
    assert sign_agreement(-agreeing, lexicon, expected) == pytest.approx(0.0)


def test_interval_head_emits_a_positive_scale() -> None:
    head = IntervalHead(10, 4, 8)
    beta, scale = head(torch.randn(5, 10))
    assert beta.shape == scale.shape == (5, 4)
    assert bool((scale > 0).all())
    lower, upper = interval_from_scale(beta, scale, 0.90)
    assert bool((upper > lower).all())
    assert z_multiplier(0.90) == pytest.approx(1.6449, abs=1e-3)


def test_coverage_falls_when_the_interval_narrows() -> None:
    generator = torch.Generator().manual_seed(9)
    beta = torch.randn(300, 3, generator=generator)
    targets = beta + torch.randn(300, 3, generator=generator) * 0.2
    scale = torch.full_like(beta, 0.2)
    nominally = report_intervals(*interval_from_scale(beta, scale, 0.90), targets, 0.90)
    narrow = report_intervals(*interval_from_scale(beta, scale * 0.3, 0.90), targets, 0.90)
    shift = coverage_shift(nominally, narrow)
    assert narrow.empirical_coverage < nominally.empirical_coverage
    assert shift["coverage_delta"] < 0.0
    assert shift["width_delta"] < 0.0


def test_the_assembled_model_returns_every_head(lexicon) -> None:
    model = Prism(
        ModelConfig(
            streams={
                "ct": StreamConfig("ct", (1, 2, 4, 4), 8, 1),
                "labs": StreamConfig("labs", (6,), 8, 1),
            },
            fused_dim=16,
            representation_dim=16,
            probe_hidden=8,
            interval_hidden=8,
            decoder_hidden=8,
            site_classifier_hidden=8,
        ),
        lexicon,
    )
    assert model.decoder is not None
    assert model.interval is not None
    assert model.site_classifier is None
    assert set(model.component_names()) == {
        "alignment",
        "invariance",
        "programme",
        "interval",
        "adversarial",
    }


def test_the_adversarial_arm_builds_a_site_classifier(lexicon) -> None:
    model = Prism(
        ModelConfig(
            streams={"labs": StreamConfig("labs", (6,), 8, 1)},
            fused_dim=8,
            representation_dim=8,
            probe_hidden=8,
            interval_hidden=8,
            decoder_hidden=8,
            site_classifier_hidden=8,
        ),
        lexicon,
        arm="adversarial",
        n_sites=4,
    )
    classifier = model.site_classifier
    assert classifier is not None
    assert classifier.n_sites == 4
    logits = classifier(torch.randn(5, 8))
    assert logits.shape == (5, 4)
    assert SiteClassifier(8, 4, 8).n_sites == 4


def test_the_reversed_gradient_is_exactly_negated() -> None:
    representation = torch.randn(4, 3, requires_grad=True)
    plain = representation.detach().clone().requires_grad_(True)
    head = torch.nn.Linear(3, 2)
    (head(plain) ** 2).sum().backward()
    (head(reverse_gradient(representation, 1.0)) ** 2).sum().backward()
    plain_gradient = plain.grad
    reversed_gradient = representation.grad
    assert plain_gradient is not None and reversed_gradient is not None
    assert torch.allclose(plain_gradient, -reversed_gradient, atol=1e-6)
    assert torch.allclose(reverse_gradient(representation.detach(), 1.0), representation.detach())


def test_a_modality_absent_from_the_backbone_is_refused(lexicon) -> None:
    with pytest.raises(ConfigError):
        Prism(
            ModelConfig(
                streams={"fluoroscopy": StreamConfig("fluoroscopy", (6,), 8, 1)},
                fused_dim=8,
                representation_dim=8,
                probe_hidden=8,
                interval_hidden=8,
                decoder_hidden=8,
                site_classifier_hidden=8,
            ),
            lexicon,
        )
    assert "ct" in MODALITY_STREAMS
    assert np.isfinite(1.0)
