"""Modelling: per-modality encoders, fusion, heads and the constrained assembly."""

from __future__ import annotations

from prism.modelling.assembly import (
    Prism,
    PrismOutput,
    SiteClassifier,
    reverse_gradient,
)
from prism.modelling.backbone import (
    PrismBackbone,
    TensorBatch,
    availability_from_lists,
    draw_plan_for_batch,
    stack_batches,
)
from prism.modelling.fusion import RepresentationFusion
from prism.modelling.interval_head import (
    IntervalHead,
    IntervalReport,
    coverage_shift,
    interval_from_scale,
    interval_width,
    leave_one_site_out,
    report_intervals,
    z_multiplier,
)
from prism.modelling.modality_dropout import (
    DropoutPlan,
    availability_tensor,
    coverage_histogram,
    draw_dropout_plan,
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
    ProbeFit,
    fit_probe_closed_form,
    head_gradient_norm,
    probe_from_state,
    probe_scores,
)
from prism.modelling.programme_head import (
    ProgrammeDecoder,
    SupportThreshold,
    block_norms,
    emissions_from_coefficients,
    select_support,
    sign_agreement,
    signs_for_support,
    support_sizes,
)
from prism.modelling.streams import (
    BagStream,
    StreamEncoder,
    TabularStream,
    VolumeStream,
    build_stream,
    stream_output_width,
    stream_parameters,
)

__all__ = [
    "BagStream",
    "DropoutPlan",
    "GatedPool",
    "IntervalHead",
    "IntervalReport",
    "LinearProbe",
    "MaskedAttentionPool",
    "MaxPool",
    "MeanPool",
    "OutcomeHead",
    "Prism",
    "PrismBackbone",
    "PrismOutput",
    "ProbeFit",
    "ProgrammeDecoder",
    "RepresentationFusion",
    "SiteClassifier",
    "StreamEncoder",
    "SupportThreshold",
    "TabularStream",
    "TensorBatch",
    "VolumeStream",
    "availability_from_lists",
    "availability_tensor",
    "block_norms",
    "build_pooling",
    "build_stream",
    "coverage_histogram",
    "coverage_shift",
    "draw_dropout_plan",
    "draw_plan_for_batch",
    "emissions_from_coefficients",
    "fit_probe_closed_form",
    "head_gradient_norm",
    "interval_from_scale",
    "interval_width",
    "leave_one_site_out",
    "probe_from_state",
    "probe_scores",
    "report_intervals",
    "reverse_gradient",
    "select_support",
    "sign_agreement",
    "signs_for_support",
    "stack_batches",
    "stream_output_width",
    "stream_parameters",
    "support_sizes",
    "z_multiplier",
]
