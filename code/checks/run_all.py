"""Verification driver.

Run this from the project root:

    PYTHONPATH=code python3 code/checks/run_all.py

It executes both verification layers, folds their outcomes onto the registered claims, writes the
claim map and the verification report, and finishes by writing the integrity manifest and the
resource list so that the manifest covers the reports it describes. A claim that could not be
executed is reported as blocked or not run; nothing is promoted to a pass on the strength of having
been attempted.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from checks import manuscript_audit, runtime_probes  # noqa: E402
from prism.audit.claims import anchor_inventory, claims, write_claim_map  # noqa: E402
from prism.audit.context import derived_context  # noqa: E402
from prism.audit.status import (  # noqa: E402
    BLOCKED,
    CODE_BEHAVIOUR,
    FAIL,
    MANUSCRIPT_ARITHMETIC,
    NOT_RUN,
    PASS,
    ClaimOutcome,
    VerificationReport,
    layer_status,
    overall_status,
    tally,
    write_verification_report,
)
from prism.cli.audit import run as run_integrity  # noqa: E402
from prism.provenance.digest import digest_file  # noqa: E402

PRECEDENCE = {FAIL: 3, BLOCKED: 2, PASS: 1, NOT_RUN: 0}


def fold(statuses: Sequence[str]) -> str:
    """Combine the outcomes of several checks on one claim."""
    if not statuses:
        return NOT_RUN
    return max(statuses, key=lambda status: PRECEDENCE[status])


def collect_outcomes() -> tuple[tuple[ClaimOutcome, ...], tuple[Mapping[str, object], ...]]:
    """Run both layers and turn their findings into claim outcomes."""
    arithmetic = manuscript_audit.run_manuscript_audit()
    probes = runtime_probes.run_probes()

    grouped: dict[str, list[tuple[str, str, str, Mapping[str, object]]]] = {}
    for finding in arithmetic:
        grouped.setdefault(_arithmetic_claim(finding.check), []).append(
            (
                finding.status,
                MANUSCRIPT_ARITHMETIC,
                finding.check,
                {"detail": finding.detail, **dict(finding.evidence)},
            )
        )
    for probe in probes:
        grouped.setdefault(probe.claim_id, []).append(
            (
                probe.status,
                CODE_BEHAVIOUR,
                probe.check,
                {"detail": probe.detail, **dict(probe.evidence)},
            )
        )

    known = {claim.claim_id for claim in claims()}
    unknown = sorted(set(grouped) - known)
    if unknown:
        raise SystemExit(f"the driver produced outcomes for unregistered claims: {unknown}")

    outcomes: list[ClaimOutcome] = []
    for claim_id in sorted(grouped):
        entries = grouped[claim_id]
        for status, layer, check, evidence in entries:
            block = dict(evidence)
            detail = str(block.pop("detail", ""))
            outcomes.append(
                ClaimOutcome(
                    claim_id=claim_id,
                    status=status,
                    layer=layer,
                    check=check,
                    detail=detail,
                    evidence=block,
                )
            )
    for claim in claims():
        if claim.claim_id in grouped:
            continue
        outcomes.append(
            ClaimOutcome(
                claim_id=claim.claim_id,
                status=NOT_RUN,
                layer=CODE_BEHAVIOUR,
                check="not-executed",
                detail=(
                    "no probe or arithmetic check was written for this claim; the code anchors "
                    "exist "
                    "and the claim is mapped, but nothing was executed against it"
                ),
                evidence={"executability": claim.executability},
            )
        )
    return tuple(outcomes), tuple(probe.as_dict() for probe in probes)


_ARITHMETIC_TO_CLAIM: Mapping[str, str] = {
    "prevalence-interval": "prevalence-rule-in",
    "prevalence-dependence": "prevalence-rule-in",
    "table1-partition-ordering": "table1-predictors",
    "criterion-arithmetic": "criterion-conjunctive",
    "component-synergy": "table2-super-additivity",
    "component-synergy-prose": "table2-super-additivity",
    "tier-zero-controls": "table2-ablation",
    "null-band-exemptions": "stat-null-band",
    "cross-region-retention": "table3-transfer-matrix",
    "transfer-diagonal-maximality": "table3-transfer-matrix",
    "reader-arm-ordering": "table4-reader-study",
    "rule-in-value-at-prevalence": "table4-reader-study",
    "cross-domain-advantage-prose": "table6-cross-domain",
    "scaling-saturation": "table7-scaling",
    "ensemble-reading": "table8-ensembling",
    "conditional-proxy-shift": "conditional-oxygenation",
}


def _arithmetic_claim(check: str) -> str:
    return _ARITHMETIC_TO_CLAIM.get(check, "eq1-objective")


DEV_NOTE = (
    "Every numerical hyper-parameter in this release is an engineering default: the article states "
    "that no experiment numeric value appears in its algorithm and prints none. The defaults are "
    "declared in configs/ and recorded per field in the run header, so a reader can see which "
    "values "
    "the article fixed and which it did not."
)


def deviations() -> tuple[Mapping[str, str], ...]:
    """The documented departures from the article, each with the section it departs from."""
    return (
        {
            "item": "no hyper-parameter is printed anywhere in the article",
            "paper_section": "Algorithm 1 caption; Methods Sec. 6.1",
            "handling": DEV_NOTE,
            "call_site": "prism/provenance/config.py:ORIGIN_ENGINEERING_DEFAULT",
        },
        {
            "item": "the cohort, its endpoints and its provenance have no section text",
            "paper_section": "Methods Sec. 6.2",
            "handling": (
                "the cohort assembly is reconstructed from Results Sec. 2.6 and the reported "
                "record "
                "counts, and every reconstructed field is marked as such in the claim map"
            ),
            "call_site": "prism/cohort/archive.py:read_archive",
        },
        {
            "item": "the site-to-region assignment, the metabolite vocabulary and two programme "
            "names "
            "are not printed",
            "paper_section": "Results Sec. 2.6; Methods Sec. 6.1, Sec. 6.6",
            "handling": (
                "recorded as reconstructions with LOW confidence in the claim map, rather than "
                "presented as reported quantities"
            ),
            "call_site": "prism/cohort/strata.py:DEFAULT_REGISTRY",
        },
        {
            "item": "the decoder section is cross-referenced by a section number the article "
            "does not "
            "contain",
            "paper_section": "Methods Sec. 6.4",
            "handling": "read as Methods Sec. 6.6, where the decoder is defined",
            "call_site": "prism/objectives/programme.py",
        },
        {
            "item": "the ablation caption exempts only the interval-head row from the null band",
            "paper_section": "Table 2 caption",
            "handling": (
                "the arithmetic check reports two further rows inside the band; the discrepancy is "
                "left standing and reported rather than smoothed"
            ),
            "call_site": "code/checks/manuscript_audit.py:check_null_band_claim",
        },
        {
            "item": "the transfer-matrix caption reads a diagonal that is not uniformly the "
            "maximum",
            "paper_section": "Table 3 Panel A caption",
            "handling": (
                "the arithmetic check finds every diagonal at the row maximum and the pooled row "
                "below every diagonal; reported as a discrepancy"
            ),
            "call_site": "code/checks/manuscript_audit.py:check_diagonal_maximality",
        },
        {
            "item": "the scaling prose places the maximum external performance at the smallest "
            "fitting fraction",
            "paper_section": "Results Sec. 3.2; Table 7",
            "handling": "the arithmetic check finds the maximum at the largest fraction; reported",
            "call_site": "code/checks/manuscript_audit.py:check_scaling_claims",
        },
        {
            "item": "the cross-domain prose prints a third advantage the table does not carry",
            "paper_section": "Results Sec. 3.1; Table 6",
            "handling": "both sequences are reported side by side",
            "call_site": "code/checks/manuscript_audit.py:check_cross_domain_prose",
        },
        {
            "item": "the cross-domain prose prints a third advantage the table does not carry",
            "paper_section": "Results Sec. 3.1; Table 6",
            "handling": "both sequences are reported side by side",
            "call_site": "code/checks/manuscript_audit.py:check_cross_domain_prose",
        },
        {
            "item": "the compute profile is absent",
            "paper_section": "no compute statement in the article",
            "handling": (
                "the release reports its own measured footprint and does not present an "
                "estimate as "
                "the study's"
            ),
            "call_site": "README.md",
        },
    )


@dataclass(frozen=True)
class DriverReport:
    """What the driver wrote, for the console line at the end."""

    overall: str
    code_status: str
    manuscript_status: str
    tally: Mapping[str, int]
    claim_to_code: Path
    verification: Path
    manifest: Path
    dataset_urls: Path


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    outcomes, probe_records = collect_outcomes()
    report = VerificationReport(
        project="prism",
        outcomes=outcomes,
        environment=runtime_probes.environment_fingerprint(),
        notes=(
            "the clinical arm, the silent exposure sample and the in-vitro line panel are not "
            "redistributable and were not observed; claims that depend on them are reported as "
            "blocked rather than as passing",
            "the clinical arm is not obtainable here, so the tables the article reports were not "
            "produced; the code layer was exercised on fixture records built by the harness, and "
            "no value measured on them is a study result",
            "the manuscript-arithmetic layer is scored separately from the code layer so that a "
            "contradiction inside the article is not read as a broken release",
        ),
    )

    per_claim: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[ClaimOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.claim_id, []).append(outcome)
    for claim_id, entries in grouped.items():
        folded = fold([entry.status for entry in entries])
        per_claim[claim_id] = {
            "status": folded,
            "checks": [entry.check for entry in entries],
            "layers": sorted({entry.layer for entry in entries}),
            "detail": "; ".join(f"{entry.check}: {entry.status}" for entry in entries),
        }

    claim_to_code = write_claim_map(
        PROJECT_ROOT / "claim_to_code.json",
        context=derived_context(),
        verification=per_claim,
        extra={
            "layer_status": {
                "code": layer_status(outcomes, CODE_BEHAVIOUR),
                "manuscript_arithmetic": layer_status(outcomes, MANUSCRIPT_ARITHMETIC),
            },
            "overall_status": overall_status(outcomes),
            "probe_outcomes": list(probe_records),
            "deviations": list(deviations()),
            "anchor_inventory": list(anchor_inventory()),
            "claim_count": len(claims()),
        },
    )
    verification_path = write_verification_report(PROJECT_ROOT / "verification_report.json", report)
    hygiene, manifest_path, urls_path = run_integrity(PROJECT_ROOT)

    report_body = json.loads(verification_path.read_text(encoding="utf-8"))
    report_body["hygiene"] = dict(hygiene.by_check())
    report_body["artefacts"] = {
        "claim_to_code_sha256": digest_file(claim_to_code),
        "integrity_manifest_sha256": digest_file(manifest_path),
        "dataset_urls_sha256": digest_file(urls_path),
    }
    report_body["tally_by_layer"] = {
        "code": tally([o for o in outcomes if o.layer == CODE_BEHAVIOUR]),
        "manuscript_arithmetic": tally([o for o in outcomes if o.layer == MANUSCRIPT_ARITHMETIC]),
    }
    verification_path.write_text(
        json.dumps(report_body, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = DriverReport(
        overall=str(report_body["overall_status"]),
        code_status=str(report_body["code_status"]),
        manuscript_status=str(report_body["manuscript_arithmetic_status"]),
        tally=dict(report_body["tally"]),
        claim_to_code=claim_to_code,
        verification=verification_path,
        manifest=manifest_path,
        dataset_urls=urls_path,
    )
    print(
        json.dumps(
            {
                "overall": summary.overall,
                "code": summary.code_status,
                "manuscript_arithmetic": summary.manuscript_status,
                "tally": summary.tally,
                "hygiene_findings": dict(hygiene.by_check()),
            },
            sort_keys=True,
        )
    )
    return 0 if summary.overall in {"VERIFIED", "PARTIALLY_VERIFIED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
