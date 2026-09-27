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
from collections.abc import Sequence
from pathlib import Path

from prism.audit.claims import claims
from prism.audit.context import derived_context
from prism.audit.integrity import HygieneReport, build_manifest, run_hygiene, write_dataset_urls
from prism.provenance.journal import get_logger
from prism.provenance.paths import project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-audit",
        description="Run the release hygiene pass and write the manifest and the resource list.",
    )
    parser.add_argument("--root", default=None, help="checkout root; defaults to the project root")
    return parser


def run(root: Path) -> tuple[HygieneReport, Path, Path]:
    package_root = root / "code" / "prism"
    report = run_hygiene(root, package_root)
    urls_path = write_dataset_urls(root / "dataset_urls.txt")
    manifest_path = root / "integrity_manifest.json"
    written, entries = build_manifest(
        root,
        manifest_path,
        report,
        extra={
            "project": "prism",
            "claim_count": len(claims()),
            "context": derived_context(),
        },
        exclude=(
            root / "claim_to_code.json",
            root / "verification_report.json",
            root / "dataset_urls.txt",
        ),
    )
    return report, written, urls_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else project_root()
    report, manifest, urls_path = run(root)
    logger = get_logger("cli.audit")
    logger.info("hygiene findings: %s", dict(report.by_check()))
    logger.info("manifest written to %s", manifest.name)
    logger.info("resource list written to %s", urls_path.name)
    print(json.dumps({"findings": dict(report.by_check()), "clean": report.clean()}))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
