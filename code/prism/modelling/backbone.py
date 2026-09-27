"""The multimodal backbone.

Ref: Methods Sec. 6.3 -- the backbone is the representation-producing half of Prism: one
encoder per stream, pooling inside each stream, and fusion after pooling. The pathway
from a record to Z = f(X) is the only thing this module owns; the heads and the
objectives are separate so that the probe stage can freeze exactly this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from prism.errors import ConfigError
from prism.modelling.fusion import RepresentationFusion
from prism.modelling.modality_dropout import (
    DropoutPlan,
    availability_tensor,
    draw_dropout_plan,
)
from prism.modelling.streams import StreamEncoder, build_stream, stream_output_width
from prism.notation import MODALITY_STREAMS
from prism.provenance.config import StreamConfig


@dataclass
class TensorBatch:
    """One mini-batch of observations in tensor form."""

    streams: Mapping[str, Tensor]
    present: Tensor
    masks: Mapping[str, Tensor] = field(default_factory=dict)
    record_ids: tuple[str, ...] = ()

    def size(self) -> int:
        return int(self.present.shape[0])

    def visible_streams(self) -> tuple[str, ...]:
        mask = self.present.any(dim=0)
        return tuple(name for column, name in enumerate(MODALITY_STREAMS) if bool(mask[column]))

    def restrict(self, plan: DropoutPlan) -> TensorBatch:
        """Apply a dropout plan, keeping the original availability for bookkeeping."""
        if plan.present.shape != self.present.shape:
            raise ConfigError("a dropout plan must match the batch it is applied to")
        return TensorBatch(
            streams=self.streams,
            present=plan.present,
            masks=self.masks,
            record_ids=self.record_ids,
        )


def stack_batches(batches: list[TensorBatch]) -> TensorBatch:
    """Concatenate mini-batches along the observation axis."""
    if not batches:
        raise ConfigError("no batch to stack")
    keys = set(batches[0].streams)
    for batch in batches[1:]:
        if set(batch.streams) != keys:
            raise ConfigError("stacked batches must carry the same streams")
    merged = {name: torch.cat([batch.streams[name] for batch in batches], dim=0) for name in keys}
    present = torch.cat([batch.present for batch in batches], dim=0)
    mask_keys = set(batches[0].masks)
    masks: dict[str, Tensor] = {}
    for name in mask_keys:
        masks[name] = torch.cat([batch.masks[name] for batch in batches], dim=0)
    ids: list[str] = []
    for batch in batches:
        ids.extend(batch.record_ids)
    return TensorBatch(streams=merged, present=present, masks=masks, record_ids=tuple(ids))


class PrismBackbone(nn.Module):
    """Per-stream encoders plus representation-level fusion."""

    streams: nn.ModuleDict
    fusion: RepresentationFusion
    order: tuple[str, ...]
    widths: dict[str, int]
    _encoders: dict[str, StreamEncoder]

    def __init__(
        self,
        stream_configs: Mapping[str, StreamConfig],
        fused_dim: int,
        *,
        presence_gate: bool = True,
    ) -> None:
        super().__init__()
        if not stream_configs:
            raise ConfigError("the backbone needs at least one stream")
        self.order = tuple(sorted(stream_configs))
        unknown = [name for name in self.order if name not in MODALITY_STREAMS]
        if unknown:
            raise ConfigError(f"unrecognised modalities in the backbone: {unknown}")
        encoders: dict[str, StreamEncoder] = {
            name: build_stream(
                name,
                config.input_shape,
                config.width,
                config.depth,
                config.dropout,
            )
            for name, config in stream_configs.items()
        }
        self._encoders = encoders
        self.streams = nn.ModuleDict(dict(encoders))
        self.widths = {name: stream_output_width(encoder) for name, encoder in encoders.items()}
        self.fusion = RepresentationFusion(self.widths, fused_dim, use_presence_gate=presence_gate)
        self.fused_dim = fused_dim

    def encode_stream(self, name: str, batch: TensorBatch) -> Tensor:
        """Encode one stream, returning zeros where the observation does not carry it."""
        encoder = self._encoders[name]
        if name not in batch.streams:
            return torch.zeros(
                (batch.size(), self.widths[name]),
                dtype=self.fusion.projection.weight.dtype,
                device=batch.present.device,
            )
        mask = batch.masks.get(name)
        encoded: Tensor = encoder(batch.streams[name], mask)
        return encoded

    def forward(self, batch: TensorBatch) -> Tensor:
        """Z = f(X) for one mini-batch."""
        if batch.present.shape[1] != len(self.order):
            raise ConfigError("the presence tensor must cover the backbone's streams")
        encoded: dict[str, Tensor] = {}
        for name in self.order:
            if name in batch.streams:
                encoded[name] = self.encode_stream(name, batch)
        fused: Tensor = self.fusion(encoded, batch.present)
        return fused

    def rep_size(self) -> int:
        return self.fused_dim

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def freeze(self) -> None:
        """Stage 1 freezes this module while the heads are fitted."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)


def draw_plan_for_batch(
    availability: Tensor, probability: float, generator: torch.Generator | None = None
) -> DropoutPlan:
    return draw_dropout_plan(availability, probability, generator)


def availability_from_lists(lists: list[tuple[str, ...]]) -> Tensor:
    return availability_tensor(lists)
