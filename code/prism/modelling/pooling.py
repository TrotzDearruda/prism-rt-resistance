"""Pooling of a modality's local blocks into one vector per observation.

Ref: Methods Sec. 6.3 -- fusion happens after the per-modality blocks are pooled, so
pooling is where a variable-size slide or volume becomes a fixed-size row.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from prism.errors import ConfigError


class MaskedAttentionPool(nn.Module):
    """Attention pooling with a mask, so an absent block never attracts weight."""

    query: Tensor
    score: nn.Linear

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 1:
            raise ConfigError("attention pooling needs a positive width")
        self.width = width
        self.query = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.query, std=1.0 / float(width) ** 0.5)
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, blocks: Tensor, mask: Tensor | None = None) -> Tensor:
        """``blocks`` is (batch, n_blocks, width); ``mask`` is (batch, n_blocks)."""
        if blocks.dim() != 3:
            raise ConfigError("attention pooling expects a three-dimensional block tensor")
        projected: Tensor = torch.tanh(self.score(blocks)).squeeze(-1)
        bias: Tensor = blocks @ self.query
        logits: Tensor = projected + bias
        if mask is not None:
            if mask.shape != logits.shape:
                raise ConfigError("attention mask must match the block dimensions")
            negative = torch.finfo(logits.dtype).min
            logits = torch.where(mask, logits, torch.full_like(logits, negative))
        weights: Tensor = torch.softmax(logits, dim=-1)
        if mask is not None:
            empty = ~mask.any(dim=-1)
            if bool(empty.any()):
                weights = torch.where(empty.unsqueeze(-1), torch.zeros_like(weights), weights)
        pooled: Tensor = torch.einsum("bn,bnd->bd", weights, blocks)
        return pooled


class MeanPool(nn.Module):
    """Masked mean over blocks, the fallback when a modality has no attention budget."""

    def forward(self, blocks: Tensor, mask: Tensor | None = None) -> Tensor:
        if blocks.dim() != 3:
            raise ConfigError("mean pooling expects a three-dimensional block tensor")
        if mask is None:
            return blocks.mean(dim=1)
        weights = mask.to(blocks.dtype).unsqueeze(-1)
        total: Tensor = (blocks * weights).sum(dim=1)
        count: Tensor = weights.sum(dim=1).clamp(min=1.0)
        return total / count


class MaxPool(nn.Module):
    """Masked maximum over blocks, used where a single focal block carries the signal."""

    def forward(self, blocks: Tensor, mask: Tensor | None = None) -> Tensor:
        if blocks.dim() != 3:
            raise ConfigError("max pooling expects a three-dimensional block tensor")
        if mask is None:
            values: Tensor = blocks.max(dim=1).values
            return values
        negative = torch.finfo(blocks.dtype).min
        masked = torch.where(mask.unsqueeze(-1), blocks, torch.full_like(blocks, negative))
        pooled: Tensor = masked.max(dim=1).values
        return torch.where(mask.any(dim=-1).unsqueeze(-1), pooled, torch.zeros_like(pooled))


class GatedPool(nn.Module):
    """Gated sum of mean and attention pooling, chosen per channel."""

    gate: nn.Linear
    attention: MaskedAttentionPool
    mean: MeanPool

    def __init__(self, width: int) -> None:
        super().__init__()
        self.attention = MaskedAttentionPool(width)
        self.mean = MeanPool()
        self.gate = nn.Linear(width, width)

    def forward(self, blocks: Tensor, mask: Tensor | None = None) -> Tensor:
        attended: Tensor = self.attention(blocks, mask)
        averaged: Tensor = self.mean(blocks, mask)
        gate: Tensor = torch.sigmoid(self.gate(attended))
        return gate * attended + (1.0 - gate) * averaged


def build_pooling(kind: str, width: int) -> nn.Module:
    """Resolve a pooling name to a module."""
    if kind == "attention":
        return MaskedAttentionPool(width)
    if kind == "mean":
        return MeanPool()
    if kind == "max":
        return MaxPool()
    if kind == "gated":
        return GatedPool(width)
    raise ConfigError(f"unknown pooling kind {kind!r}")
