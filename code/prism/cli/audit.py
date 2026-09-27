"""Audit entry point.

Ref: the release asks a reviewer to check three things by hand: that the article's statements are
each answerable in the code, that no shipped file records a machine-specific path or a
generated-content phrase, and that the manifest lists exactly the files that are present. This entry
point writes the resource list and the integrity manifest; the claim map and the verification report
are written by the verification driver, which is the only thing that can fill in execution outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from prism.audit.claims import claims
from prism.audit.context import derived_context
from prism.audit.integrity import (
    HygieneReport,
    build_manifest,
    run_hygiene,
    write_dataset_urls,
)
from prism.provenance.journal import get_logger
from prism.provenance.paths import project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-audit",
        description="Run the release hygiene pass and write the manifest and the resource list.",
    )
    parser.add_argument("--root", default=None, help="checkout root; defaults to the project root")
    return parser


def audit_hygiene(root: Path) -> HygieneReport:
    """Run the release hygiene pass over a checkout."""
    return run_hygiene(root, root / "code" / "prism")


def write_resource_list(root: Path) -> Path:
    """Write the resource list. Separate from the manifest so the manifest can digest it."""
    return write_dataset_urls(root / "dataset_urls.txt")


def write_integrity(
    root: Path, hygiene: HygieneReport, extra: Mapping[str, object] | None = None
) -> Path:
    """Write the integrity manifest, which is the last thing a release writes.

    The manifest digests every other shipped file, so anything written after it would leave the
    manifest describing a tree that no longer exists. It is therefore its own final step and it
    excludes only itself.
    """
    manifest_path = root / "integrity_manifest.json"
    body: dict[str, object] = {"project": "prism", "claim_count": len(claims())}
    body.update(dict(extra or {}))
    written, _ = build_manifest(root, manifest_path, hygiene, extra=body, exclude=())
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else project_root()
    report = audit_hygiene(root)
    urls_path = write_resource_list(root)
    manifest = write_integrity(root, report, {"context": derived_context()})
    logger = get_logger("cli.audit")
    logger.info("hygiene findings: %s", dict(report.by_check()))
    logger.info("manifest written to %s", manifest.name)
    logger.info("resource list written to %s", urls_path.name)
    print(json.dumps({"findings": dict(report.by_check()), "clean": report.clean()}))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
