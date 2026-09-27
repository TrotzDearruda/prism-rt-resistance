"""Per-modality encoders.

Ref: Methods Sec. 6.3 -- the observation carries a pre-treatment pelvic volume, a
histopathology block and routine laboratory values; the histology unit follows the
whole-slide decoders the study builds on, and the imaging unit is the convolutional
family kept as the architectural comparison. Each encoder maps its own stream to a
vector of the shared width before fusion.
"""

from __future__ import annotations

from torch import Tensor, nn

from prism.errors import ConfigError
from prism.modelling.pooling import build_pooling


def _activation() -> nn.Module:
    return nn.GELU()


class StreamEncoder(nn.Module):
    """Common surface of a stream encoder: a fixed-width row per observation."""

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class VolumeStream(StreamEncoder):
    """Convolutional encoder over a three-dimensional pre-treatment volume."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
        *,
        width_multiplier: float = 1.6,
    ) -> None:
        super().__init__()
        if in_channels < 1 or width < 1 or depth < 1:
            raise ConfigError("a volume stream needs positive channels, width and depth")
        layers: list[nn.Module] = []
        channels = in_channels
        for level in range(depth):
            out_channels = max(4, int(round(width * (width_multiplier**level))))
            stride = 2 if level > 0 else 1
            layers.append(
                nn.Conv3d(
                    channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.GroupNorm(_groups(out_channels), out_channels))
            layers.append(_activation())
            if dropout > 0.0:
                layers.append(nn.Dropout3d(dropout))
            channels = out_channels
        self.blocks = nn.Sequential(*layers)
        self.projection = nn.Linear(channels, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, volume: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            raise ConfigError("a volume stream reads a dense grid and takes no block mask")
        if volume.dim() != 5:
            raise ConfigError("a volume stream expects (batch, channels, depth, height, width)")
        features: Tensor = self.blocks(volume)
        spaced: Tensor = features.flatten(start_dim=2).mean(dim=-1)
        projected: Tensor = self.projection(spaced)
        normalised: Tensor = self.norm(projected)
        return normalised


class BagStream(StreamEncoder):
    """Set encoder over the patch features of one histopathology block."""

    pool: nn.Module
    patch: nn.Sequential
    projection: nn.Linear
    norm: nn.LayerNorm

    def __init__(
        self,
        patch_dim: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
        *,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        if patch_dim < 1 or width < 1 or depth < 1:
            raise ConfigError("a bag stream needs positive patch dimension, width and depth")
        layers: list[nn.Module] = [nn.Linear(patch_dim, width), _activation()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(_activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.patch = nn.Sequential(*layers)
        self.pool = build_pooling(pooling, width)
        self.projection = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, patches: Tensor, mask: Tensor | None = None) -> Tensor:
        if patches.dim() != 3:
            raise ConfigError("a bag stream expects (batch, patches, patch_dim)")
        embedded: Tensor = self.patch(patches)
        pooled: Tensor = self.pool(embedded, mask)
        projected: Tensor = self.projection(pooled)
        normalised: Tensor = self.norm(projected)
        return normalised


class TabularStream(StreamEncoder):
    """Encoder over routine laboratory values, with a small non-linearity budget."""

    def __init__(self, n_features: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_features < 1 or width < 1 or depth < 1:
            raise ConfigError("a tabular stream needs positive features, width and depth")
        layers: list[nn.Module] = []
        input_dim = n_features
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, width))
            layers.append(_activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = width
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, table: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            raise ConfigError("a tabular stream reads a single row and takes no block mask")
        if table.dim() != 2:
            raise ConfigError("a tabular stream expects (batch, features)")
        encoded: Tensor = self.mlp(table)
        normalised: Tensor = self.norm(encoded)
        return normalised


def _groups(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


def build_stream(
    modality: str,
    input_shape: tuple[int, ...],
    width: int,
    depth: int,
    dropout: float = 0.0,
) -> StreamEncoder:
    """Resolve a modality name to the encoder that reads its stream shape."""
    if modality in {"ct", "mri"}:
        if len(input_shape) != 4:
            raise ConfigError(
                f"{modality} expects a per-record volume shape of channels by depth by height "
                f"by width"
            )
        return VolumeStream(in_channels=input_shape[0], width=width, depth=depth, dropout=dropout)
    if modality == "histology":
        if len(input_shape) == 1:
            return TabularStream(
                n_features=input_shape[0], width=width, depth=depth, dropout=dropout
            )
        if len(input_shape) == 2:
            return BagStream(patch_dim=input_shape[1], width=width, depth=depth, dropout=dropout)
        raise ConfigError("histology expects a patch dimension or a flat feature vector")
    if modality == "labs":
        if len(input_shape) != 1:
            raise ConfigError("labs expects a one-dimensional feature shape")
        return TabularStream(n_features=input_shape[0], width=width, depth=depth, dropout=dropout)
    raise ConfigError(f"no stream encoder is registered for {modality!r}")


def stream_output_width(module: StreamEncoder) -> int:
    """Read back the width an encoder emits, used when fusing heterogeneous streams."""
    if isinstance(module, (VolumeStream, BagStream, TabularStream)):
        return int(module.norm.normalized_shape[0])
    raise ConfigError(f"cannot read a width from {type(module).__name__}")


def stream_parameters(module: StreamEncoder) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
