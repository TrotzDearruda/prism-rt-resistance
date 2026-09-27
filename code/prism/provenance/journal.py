"""Run journal.

Ref: Reporting requirements in Methods Sec. 6.8 -- the analysis plan is fixed before
any held-out partition is inspected, so every run writes its configuration and its
partition digests to disk before the first optimisation step.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

LOGGER_NAME = "prism"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_logger(suffix: str = "") -> logging.Logger:
    """A package logger, never the root logger."""
    name = f"{LOGGER_NAME}.{suffix}" if suffix else LOGGER_NAME
    return logging.getLogger(name)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Attach a single stream handler, plus a file handler when a path is given."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(log_file, encoding="utf-8")
        to_file.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(to_file)
    logger.propagate = False


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def record_run_header(
    out_dir: Path,
    run_name: str,
    config: Any,
    partitions: Mapping[str, int],
    seeds: tuple[int, ...],
) -> Path:
    """Persist the run header before any fitting happens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "run": run_name,
        "config": _plain(config),
        "partition_sizes": {str(k): int(v) for k, v in partitions.items()},
        "seeds": list(seeds),
    }
    target = out_dir / "run_header.json"
    target.write_text(json.dumps(header, indent=2, sort_keys=True), encoding="utf-8")
    get_logger("journal").info(
        "run %s declared with %d partitions and %d ensemble members",
        run_name,
        len(partitions),
        len(seeds),
    )
    return target


def record_metric(out_dir: Path, name: str, value: float) -> None:
    """Append one scalar to the run's metric stream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = out_dir / "metrics.jsonl"
    line = json.dumps({"metric": name, "value": float(value)}, sort_keys=True)
    with stream.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
