"""Command line entry points: train, evaluate, emit, report and audit."""

from __future__ import annotations

from prism.cli._shared import (
    CohortContext,
    add_cohort_argument,
    add_common_arguments,
    archive_path,
    as_float,
    config_directory,
    data_directory,
    describe_config,
    entry,
    load_cohort,
    parse_arguments,
    resolve_config,
    resolve_device,
    resolve_lexicon,
    resolve_paths,
)

__all__ = [
    "CohortContext",
    "add_common_arguments",
    "add_cohort_argument",
    "archive_path",
    "as_float",
    "config_directory",
    "data_directory",
    "describe_config",
    "entry",
    "load_cohort",
    "parse_arguments",
    "resolve_config",
    "resolve_device",
    "resolve_lexicon",
    "resolve_paths",
]
