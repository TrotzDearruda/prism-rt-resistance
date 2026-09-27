"""Content digests.

Checkpoints record the digest of the state they carry so a resume can tell a truncated
file from a complete one, and the release manifest records the digest of every shipped
source file. The manifest path is excluded from its own inventory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from prism.errors import ResourceError

# Names skipped wherever they appear: build and cache directories that never ship.
SKIP_ANYWHERE = frozenset(
    {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".venv",
        "node_modules",
    }
)
# Names skipped only at the top level of the tree. A dataset root and an artefact root sit beside
# the release, but a directory deeper in the tree that happens to carry the same name is release
# content -- a configuration group called ``data`` was silently dropped from the manifest by an
# unanchored rule, which is the same failure an unanchored ``.gitignore`` entry produces.
SKIP_AT_ROOT = frozenset({"artifacts", "data", ".cache"})
SKIP_SUFFIXES = frozenset({".pyc", ".pyo", ".DS_Store", ".ipynb_checkpoints"})

DIGEST_CHUNK = 1 << 20


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    if not path.is_file():
        raise ResourceError(f"cannot digest a file that is not there: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DIGEST_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class InventoryEntry:
    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _skipped(relative: tuple[str, ...]) -> bool:
    if any(part in SKIP_ANYWHERE for part in relative):
        return True
    return bool(relative) and relative[0] in SKIP_AT_ROOT


def iter_source_files(root: Path, suffixes: Sequence[str] | None = None) -> list[Path]:
    """Every regular file under ``root`` that is not a build or cache artefact."""
    allowed = set(suffixes) if suffixes is not None else None
    found: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.parts[len(root.parts) :]
        if _skipped(relative):
            continue
        if candidate.suffix in SKIP_SUFFIXES or candidate.name in SKIP_SUFFIXES:
            continue
        if allowed is not None and candidate.suffix not in allowed:
            continue
        found.append(candidate)
    return found


def inventory(
    root: Path,
    *,
    suffixes: Sequence[str] | None = None,
    exclude: Iterable[Path] = (),
) -> list[InventoryEntry]:
    """Digest every shipped file under ``root`` except the excluded paths."""
    banned = {path.resolve() for path in exclude}
    entries: list[InventoryEntry] = []
    for path in iter_source_files(root, suffixes):
        if path.resolve() in banned:
            continue
        entries.append(
            InventoryEntry(
                relative_path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=digest_file(path),
            )
        )
    return entries


def inventory_digest(entries: Sequence[InventoryEntry]) -> str:
    """A single digest over an ordered inventory, so the manifest commits to itself."""
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(entry.relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(entry.sha256.encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def write_manifest(
    path: Path,
    entries: Sequence[InventoryEntry],
    extra: Mapping[str, object],
) -> Path:
    """Write the manifest, keeping its own path out of the recorded inventory."""
    payload = {
        "entry_count": len(entries),
        "inventory_digest": inventory_digest(entries),
        "entries": [entry.as_dict() for entry in entries],
    }
    payload.update({str(key): value for key, value in extra.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
