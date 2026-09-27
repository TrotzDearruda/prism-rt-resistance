"""Layout discovery.

Every path the package writes is derived from the installed location of this file, so a
checkout works from any working directory and no machine-specific prefix is recorded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

ENV_ROOT_OVERRIDE = "PRISM_PROJECT_ROOT"
ENV_ARTIFACT_OVERRIDE = "PRISM_ARTIFACT_ROOT"


def project_root() -> Path:
    """Root of the checkout, overridable for a read-only install tree."""
    override = os.environ.get(ENV_ROOT_OVERRIDE)
    return Path(override).resolve() if override else PROJECT_ROOT


def config_root() -> Path:
    return project_root() / "configs"


def default_artifact_root() -> Path:
    override = os.environ.get(ENV_ARTIFACT_OVERRIDE)
    if override:
        return Path(override).resolve()
    return project_root() / "artifacts"


@dataclass(frozen=True)
class RunPaths:
    """Directory bundle for one run: checkpoints, logs, tables and emissions."""

    root: Path
    run_name: str

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def emissions(self) -> Path:
        return self.root / "emissions"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    def create(self) -> RunPaths:
        for directory in (
            self.root,
            self.checkpoints,
            self.logs,
            self.tables,
            self.emissions,
            self.predictions,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def checkpoint_file(self, member: int, *, best: bool = False) -> Path:
        """Checkpoint path for one ensemble member.

        The plain path always holds the most recent state, which is what a resume reads; the
        ``best`` path holds the state at the epoch that scored highest on the development split,
        which is what a table reports.
        """
        suffix = ".best.pt" if best else ".pt"
        return self.checkpoints / f"{self.run_name}.member{member:02d}{suffix}"

    def table_file(self, table: str, suffix: str = "json") -> Path:
        return self.tables / f"{table}.{suffix}"


def run_paths(run_name: str, artifact_root: Path | None = None) -> RunPaths:
    base = artifact_root if artifact_root is not None else default_artifact_root()
    return RunPaths(root=base / run_name, run_name=run_name)


def data_root() -> Path:
    """Where public resources are staged by ``scripts/prepare_data.sh``."""
    return project_root() / "data"


def cache_root() -> Path:
    return project_root() / ".cache"
