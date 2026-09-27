"""Representation-level fusion.

Ref: Methods Sec. 6.3 -- fusion happens at the representation stage, after the modality
blocks are pooled. A site that cannot acquire a modality supplies zeros for it, and the
fusion layer is told which rows are present so that an absent stream does not shift the
fused vector for a different reason from the one it should.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from prism.errors import ConfigError
from prism.notation import MODALITY_STREAMS


class RepresentationFusion(nn.Module):
    """Concatenate present streams, mark the absent ones, project to the fused width."""

    projection: nn.Linear
    gate: nn.Linear
    norm: nn.LayerNorm

    def __init__(
        self,
        stream_widths: Mapping[str, int],
        fused_dim: int,
        *,
        use_presence_gate: bool = True,
    ) -> None:
        super().__init__()
        if not stream_widths:
            raise ConfigError("fusion needs at least one stream width")
        self.order: tuple[str, ...] = tuple(sorted(stream_widths))
        self.stream_widths: dict[str, int] = dict(stream_widths)
        total = sum(self.stream_widths.values()) + len(self.order)
        self.use_presence_gate = use_presence_gate
        self.projection = nn.Linear(total, fused_dim)
        self.gate = nn.Linear(len(self.order), len(self.order))
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, streams: Mapping[str, Tensor], present: Tensor) -> Tensor:
        """``present`` is (batch, n_streams) in ``self.order`` order."""
        if present.dim() != 2 or present.shape[1] != len(self.order):
            raise ConfigError("the presence tensor must be (batch, n_streams)")
        batch = int(present.shape[0])
        working_dtype = self.projection.weight.dtype
        blocks: list[Tensor] = []
        for name in self.order:
            width = self.stream_widths[name]
            if name in streams:
                block = streams[name]
                if block.shape != (batch, width):
                    raise ConfigError(
                        f"stream {name} carries shape {tuple(block.shape)}, expected "
                        f"({batch}, {width})"
                    )
            else:
                block = torch.zeros((batch, width), dtype=working_dtype, device=present.device)
            blocks.append(block)
        indicator = present.to(working_dtype)
        if self.use_presence_gate:
            gate = torch.sigmoid(self.gate(indicator))
            blocks = [block * gate[:, column : column + 1] for column, block in enumerate(blocks)]
        stacked: Tensor = torch.cat([*blocks, indicator], dim=1)
        projected: Tensor = self.projection(stacked)
        fused: Tensor = self.norm(projected)
        return fused

    def expected_streams(self) -> tuple[str, ...]:
        return tuple(name for name in self.order if name in MODALITY_STREAMS)
