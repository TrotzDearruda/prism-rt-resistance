"""Optimiser construction and gradient handling.

Ref: Algorithm 1 -- Stage 1 fits the heads with the encoders frozen and Stage 2 releases them,
so the two stages need different learning rates and the encoder group does not exist during
Stage 1. Ref: Methods Sec. 6.10 -- the run is repeated over random seeds, which means the
optimiser state has to be reconstructible from a checkpoint rather than from a stored object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from prism.errors import ConfigError

SUPPORTED_PRECISIONS: tuple[str, ...] = ("fp32", "bf16", "fp16")


@dataclass(frozen=True)
class OptimiserSettings:
    """Learning rates, decay and clipping, taken from the run configuration."""

    lr_head: float
    lr_encoder: float
    weight_decay: float
    grad_clip: float

    def validate(self) -> None:
        if self.lr_head <= 0.0 or self.lr_encoder <= 0.0:
            raise ConfigError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ConfigError("a weight decay cannot be negative")
        if self.grad_clip <= 0.0:
            raise ConfigError("a gradient clip must be positive")


def parameter_groups(
    head_parameters: Sequence[nn.Parameter],
    encoder_parameters: Sequence[nn.Parameter],
    settings: OptimiserSettings,
    *,
    include_encoder: bool,
) -> list[dict[str, Any]]:
    """One group per learning rate, with the encoder group omitted during Stage 1."""
    settings.validate()
    groups: list[dict[str, Any]] = [
        {"params": list(head_parameters), "lr": settings.lr_head, "name": "head"}
    ]
    if include_encoder:
        encoder_group = [parameter for parameter in encoder_parameters if parameter.requires_grad]
        if not encoder_group:
            raise ConfigError("the encoder group is empty but was requested")
        groups.append({"params": encoder_group, "lr": settings.lr_encoder, "name": "encoder"})
    return groups


def build_optimiser(
    head_parameters: Sequence[nn.Parameter],
    encoder_parameters: Sequence[nn.Parameter],
    settings: OptimiserSettings,
    *,
    include_encoder: bool,
) -> torch.optim.AdamW:
    groups = parameter_groups(
        head_parameters, encoder_parameters, settings, include_encoder=include_encoder
    )
    return torch.optim.AdamW(groups, lr=settings.lr_head, weight_decay=settings.weight_decay)


def clip_gradients(parameters: Sequence[nn.Parameter], max_norm: float) -> float:
    """Clip in place and return the pre-clip total norm, which is worth logging."""
    trainable = [parameter for parameter in parameters if parameter.grad is not None]
    if not trainable:
        return 0.0
    total: Tensor = torch.nn.utils.clip_grad_norm_(trainable, max_norm)
    return float(total)


def resolve_precision(name: str, device: torch.device) -> torch.dtype:
    """Map a precision name onto a dtype, refusing a choice the device cannot serve."""
    if name not in SUPPORTED_PRECISIONS:
        raise ConfigError(f"precision must be one of {SUPPORTED_PRECISIONS}, found {name!r}")
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        if device.type not in {"cuda", "cpu"}:
            raise ConfigError("bfloat16 needs a device that supports it")
        return torch.bfloat16
    if device.type != "cuda":
        raise ConfigError("float16 autocast is a CUDA path in this release")
    return torch.float16


def autocast_context(device: torch.device, precision: str) -> Any:
    """Autocast context for a configured precision, disabled for fp32."""
    dtype = resolve_precision(precision, device)
    if dtype is torch.float32:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


def scale_learning_rate(optimiser: torch.optim.Optimizer, factor: float) -> Mapping[str, float]:
    """Multiply every group rate by a factor, returning the previous rates."""
    if factor <= 0.0:
        raise ConfigError("a learning-rate factor must be positive")
    previous: dict[str, float] = {}
    for position, group in enumerate(optimiser.param_groups):
        name = str(group.get("name", f"group{position}"))
        previous[name] = float(group["lr"])
        group["lr"] = float(group["lr"]) * factor
    return previous


def gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    """L2 norm of the current gradients, for a divergence check between epochs."""
    total = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        squared: Tensor = gradient.detach().pow(2)
        total += float(squared.sum())
    return math.sqrt(total)


def describe_groups(optimiser: torch.optim.Optimizer) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "name": str(group.get("name", f"group{position}")),
            "lr": float(group["lr"]),
            "n_parameters": float(len(group["params"])),
        }
        for position, group in enumerate(optimiser.param_groups)
    )
