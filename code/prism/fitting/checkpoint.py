"""Checkpoint writing and reading.

Ref: Methods Sec. 6.10 -- the run is repeated over random seeds and the seed is part of the run
identity, so a checkpoint carries the seed that produced it and a resume restores it before any
tensor is touched. Writes are atomic: the state lands in a temporary file in the destination
directory and is moved into place, so a run interrupted mid-write leaves the previous
checkpoint intact rather than a truncated one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from prism.errors import CheckpointError
from prism.numerics.seed import SeedState, set_seed
from prism.provenance.digest import digest_bytes, digest_file

FORMAT_VERSION = 2


@dataclass(frozen=True)
class CheckpointMetadata:
    """Everything a resumed run needs to reconstruct itself."""

    run: str
    member: int
    stage: str
    epoch: int
    global_step: int
    seed: int
    deterministic: bool
    format_version: int
    state_digest: str
    metrics: Mapping[str, float]
    extra: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "run": self.run,
            "member": self.member,
            "stage": self.stage,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "format_version": self.format_version,
            "state_digest": self.state_digest,
            "metrics": dict(self.metrics),
            "extra": dict(self.extra),
        }


def model_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Digest of a state dictionary, taken in sorted key order so it is order-independent.

    Tensors are widened to double before the digest is taken, so a checkpoint written under
    one default dtype is still recognised when the same values are read back under another.
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().to(torch.float64).cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    seed_state: SeedState,
    metadata: Mapping[str, Any],
    *,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> CheckpointMetadata:
    """Write a checkpoint atomically and return the metadata that was stored."""
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_state": state,
        "model_state_digest": model_state_digest(state),
        "seed": seed_state.as_dict(),
        "metadata": dict(metadata),
    }
    if optimiser is not None:
        payload["optimiser_state"] = optimiser.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return CheckpointMetadata(
        run=str(metadata.get("run", path.stem)),
        member=int(metadata.get("member", 0)),
        stage=str(metadata.get("stage", "unknown")),
        epoch=int(metadata.get("epoch", 0)),
        global_step=int(metadata.get("global_step", 0)),
        seed=seed_state.seed,
        deterministic=seed_state.deterministic,
        format_version=FORMAT_VERSION,
        state_digest=payload["model_state_digest"],
        metrics={
            str(key): float(value) for key, value in dict(metadata.get("metrics", {})).items()
        },
        extra={str(key): value for key, value in dict(metadata.get("extra", {})).items()},
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_seed: bool = True,
) -> CheckpointMetadata:
    """Read a checkpoint, verify its digest, and restore the seed before anything else."""
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"{path} does not carry a mapping payload")
    version = int(payload.get("format_version", 0))
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"{path} was written in format {version}, this release reads {FORMAT_VERSION}"
        )
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise CheckpointError(f"{path} carries no model state")
    recorded = str(payload.get("model_state_digest", ""))
    actual = model_state_digest(state)
    if recorded and recorded != actual:
        raise CheckpointError(f"{path} failed its own state digest")
    model.load_state_dict(dict(state))

    seed_block = payload.get("seed")
    if not isinstance(seed_block, Mapping):
        raise CheckpointError(f"{path} carries no seed block")
    seed = int(seed_block.get("seed", 0))
    deterministic = bool(seed_block.get("deterministic", False))
    if restore_seed:
        set_seed(seed, deterministic=deterministic)
    if optimiser is not None and "optimiser_state" in payload:
        optimiser.load_state_dict(payload["optimiser_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return CheckpointMetadata(
        run=str(metadata.get("run", path.stem)),
        member=int(metadata.get("member", 0)),
        stage=str(metadata.get("stage", "unknown")),
        epoch=int(metadata.get("epoch", 0)),
        global_step=int(metadata.get("global_step", 0)),
        seed=seed,
        deterministic=deterministic,
        format_version=version,
        state_digest=actual,
        metrics={
            str(key): float(value) for key, value in dict(metadata.get("metrics", {})).items()
        },
        extra={str(key): value for key, value in dict(metadata.get("extra", {})).items()},
    )


def read_metadata(path: Path) -> CheckpointMetadata:
    """Read a checkpoint's metadata without touching a model."""
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"{path} does not carry a mapping payload")
    seed_block = payload.get("seed")
    seed = int(seed_block.get("seed", 0)) if isinstance(seed_block, Mapping) else 0
    deterministic = (
        bool(seed_block.get("deterministic", False)) if isinstance(seed_block, Mapping) else False
    )
    metadata = payload.get("metadata")
    block = metadata if isinstance(metadata, Mapping) else {}
    return CheckpointMetadata(
        run=str(block.get("run", path.stem)),
        member=int(block.get("member", 0)),
        stage=str(block.get("stage", "unknown")),
        epoch=int(block.get("epoch", 0)),
        global_step=int(block.get("global_step", 0)),
        seed=seed,
        deterministic=deterministic,
        format_version=int(payload.get("format_version", 0)),
        state_digest=str(payload.get("model_state_digest", "")),
        metrics={str(key): float(value) for key, value in dict(block.get("metrics", {})).items()},
        extra={str(key): value for key, value in dict(block.get("extra", {})).items()},
    )


def checkpoint_digest(path: Path) -> str:
    """Digest of the checkpoint file itself, for the release manifest."""
    return digest_file(path)


def write_metadata_sidecar(path: Path, metadata: CheckpointMetadata) -> Path:
    """A readable copy of the metadata next to the binary checkpoint.

    The binary form is the record of truth; the sidecar exists so that a table can be rebuilt
    without loading a state dictionary, and it carries the digest of the checkpoint it
    describes so the two cannot drift apart unnoticed.
    """
    sidecar = path.with_suffix(path.suffix + ".json")
    body: dict[str, Any] = dict(metadata.as_dict())
    body["checkpoint_digest"] = checkpoint_digest(path) if path.is_file() else ""
    sidecar.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def sidecar_matches(path: Path) -> bool:
    """Whether a sidecar still describes the checkpoint sitting next to it."""
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    body = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(body, Mapping):
        return False
    recorded = str(body.get("checkpoint_digest", ""))
    if not recorded:
        return False
    return recorded == digest_bytes(path.read_bytes())
