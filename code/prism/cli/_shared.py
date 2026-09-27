"""Shared helpers for the command line entry points.

Every entry point resolves its configuration the same way: read one experiment file, apply the
``key=value`` overrides, freeze the result, and then hand it to the layer that owns the work. The
helpers here exist so that the four entry points cannot drift apart in how they read a
configuration or where they write their artefacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from prism.cohort.archive import (
    default_archive_path,
    describe_archive,
    read_archive,
)
from prism.cohort.lexicon import Lexicon, default_lexicon
from prism.cohort.partitions import PartitionSpec, partition_cohort
from prism.cohort.schema import Cohort, Layer
from prism.errors import ConfigError
from prism.provenance.config import (
    ExperimentConfig,
    ModelConfig,
    load_experiment,
)
from prism.provenance.journal import configure_logging, get_logger
from prism.provenance.paths import RunPaths, config_root, data_root, run_paths


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment",
        default="main",
        help="name of the file under configs/experiment, without the extension",
    )
    parser.add_argument(
        "--config-root",
        default=None,
        help="directory holding the config groups; defaults to the checkout's configs/",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override applied after the experiment file is read",
    )
    parser.add_argument("--artifact-root", default=None, help="where the run writes its output")
    parser.add_argument("--device", default="cpu", help="torch device string")
    parser.add_argument("--log-level", default="INFO")


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    root = Path(args.config_root) if args.config_root else config_root()
    path = root / "experiment" / f"{args.experiment}.yaml"
    return load_experiment(path, args.override)


def resolve_paths(args: argparse.Namespace, config: ExperimentConfig) -> RunPaths:
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    suffix = "default" if not args.override else "override"
    return run_paths(f"{config.name}-{suffix}", artifact_root).create()


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("a CUDA device was requested but none is visible")
    return device


def archived_shapes(model: ModelConfig) -> dict[str, tuple[int, ...]]:
    """The per-stream shapes the archive must carry to match a configuration."""
    return {name: stream.input_shape for name, stream in model.streams.items()}


def resolve_lexicon(config: ExperimentConfig) -> Lexicon:
    lexicon = default_lexicon()
    if lexicon.size != config.vocabulary.size:
        raise ConfigError(
            f"the configuration declares {config.vocabulary.size} metabolites while the named "
            f"vocabulary holds {lexicon.size}"
        )
    if lexicon.programme_ids != config.vocabulary.programme_ids:
        raise ConfigError("the configured programme order does not match the vocabulary")
    return lexicon


@dataclass(frozen=True)
class CohortContext:
    """The partitioned cohort a run works on, next to how it was obtained."""

    parts: dict[Layer, Cohort]
    source: str
    lexicon: Lexicon

    def validation_partition(self) -> Cohort:
        if Layer.INTERNAL in self.parts and len(self.parts[Layer.INTERNAL]) > 0:
            return self.parts[Layer.INTERNAL]
        return self.parts[Layer.DEVELOPMENT]

    def held_out(self) -> Cohort:
        if Layer.EXTERNAL in self.parts and len(self.parts[Layer.EXTERNAL]) > 0:
            return self.parts[Layer.EXTERNAL]
        return self.parts[Layer.SILENT]


def archive_path(override: str | None) -> Path:
    """Where the staged arm is read from, and where it is expected to be."""
    if override:
        return Path(override)
    return default_archive_path(data_root())


def load_cohort(
    config: ExperimentConfig, cohort: str | None, *, seed: int | None = None
) -> CohortContext:
    """Read the arm the data owner staged and layer it.

    The clinical arm is held by the participating institutions and is not redistributable, so this
    release ships the reader rather than a cohort: a run reads the archive the data owner staged and
    reports plainly when it is absent. The partition is taken at the patient level with the sites
    the configuration names, so a staged arm is layered the same way whatever shape it arrives in.
    """
    lexicon = resolve_lexicon(config)
    resolved_seed = config.train.seed if seed is None else seed
    path = archive_path(cohort)
    staged = read_archive(path, expected_shapes=archived_shapes(config.model))
    reported_records = config.cohort.n_records + config.cohort.n_silent
    if len(staged) != reported_records:
        raise ConfigError(
            f"the staged arm carries {len(staged)} records while the configuration declares "
            f"{reported_records}; the arm a run reads has to be the one the study describes"
        )
    if set(staged.sites()) != set(config.cohort.sites):
        raise ConfigError(
            f"the staged arm carries sites {sorted(staged.sites())} while the configuration names "
            f"{sorted(config.cohort.sites)}"
        )
    held = tuple(name for name in config.cohort.sites if name in config.cohort.holdout_sites)
    developmental = tuple(
        name for name in config.cohort.sites if name not in config.cohort.holdout_sites
    )
    if not developmental:
        raise ConfigError("the configuration names no development site")
    held_patients = {record.patient_id for record in staged if record.site in set(held)}
    development_patients = {record.patient_id for record in staged if record.site not in set(held)}
    if not development_patients or not held_patients:
        raise ConfigError(
            "the staged arm must carry records at a development site and at a holdout site"
        )
    fitted = max(1, int(round(0.70 * len(development_patients))))
    spec = PartitionSpec(
        development_patients=fitted,
        internal_patients=max(1, len(development_patients) - fitted),
        external_patients=len(held_patients),
        silent_patients=0,
        development_sites=developmental,
        holdout_sites=held,
        external_regions=config.cohort.regions[-2:],
    )
    parts = partition_cohort(staged, spec, resolved_seed)
    return CohortContext(
        parts=parts, source=describe_archive(path).provenance or str(path), lexicon=lexicon
    )


def add_cohort_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cohort",
        default=None,
        help=(
            "path to the staged arm archive; defaults to data/cohort.npz, and the clinical arm is "
            "not redistributable so a run without one reports that it is absent"
        ),
    )


def entry(
    args: argparse.Namespace,
    suffix: str,
) -> tuple[ExperimentConfig, RunPaths, torch.device]:
    """Common preamble: configure logging, read the configuration, prepare the run directory."""
    configure_logging(args.log_level)
    config = resolve_config(args)
    paths = resolve_paths(args, config)
    device = resolve_device(args.device)
    get_logger(suffix).info(
        "run %s on %s with effective batch %d",
        config.name,
        device.type,
        config.effective_batch(),
    )
    return config, paths, device


def data_directory() -> Path:
    return data_root()


def config_directory() -> Path:
    return config_root()


def describe_config(config: ExperimentConfig) -> dict[str, object]:
    """A compact description of a run, printed by every entry point that starts one."""
    return {
        "name": config.name,
        "arm": config.arm,
        "invariance": config.objective.invariance_mode,
        "stages": [config.train.stage1_epochs, config.train.stage2_epochs],
        "batch_size": config.train.batch_size,
        "grad_accum": config.train.grad_accum,
        "world_size": config.train.world_size,
        "effective_batch": config.effective_batch(),
        "precision": config.train.precision,
        "ensemble_members": config.train.ensemble_members,
    }


def parse_arguments(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    return parser.parse_args(argv)


def as_float(value: object, where: str) -> float:
    """Read a numeric cell out of a table row, refusing anything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} is not a number")
    return float(value)
