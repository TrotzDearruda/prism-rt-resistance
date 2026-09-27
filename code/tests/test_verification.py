"""The claim map, the release hygiene pass and the integrity manifest."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from prism.audit.claims import (
    BLOCKED_DATA,
    DESIGN_ONLY,
    EXECUTABLE,
    NOT_REPORTED,
    anchor_inventory,
    claim_by_id,
    claims,
    claims_of_kind,
    write_claim_map,
)
from prism.audit.context import (
    ENGINEERING,
    FIELDS,
    derived_context,
    engineering_defaults,
    field_value,
    reconstructions,
)
from prism.audit.integrity import (
    FORBIDDEN_PHRASES,
    GENERATED_REPORTS,
    Finding,
    HygieneReport,
    build_manifest,
    imported_modules,
    manifest_covers,
    module_names,
    run_hygiene,
)
from prism.audit.status import (
    BLOCKED,
    CODE_BEHAVIOUR,
    FAIL,
    MANUSCRIPT_ARITHMETIC,
    NOT_RUN,
    PARTIALLY_VERIFIED,
    PASS,
    UNVERIFIED,
    VERIFIED,
    ClaimOutcome,
    VerificationReport,
    blocked_claims,
    failing_claims,
    layer_status,
    overall_status,
    require_execution_evidence,
    tally,
    write_verification_report,
)
from prism.errors import ConfigError, ResourceError
from prism.provenance.digest import digest_file, inventory
from prism.provenance.paths import project_root
from prism.reference.urls import (
    ANNOTATED_SERIES,
    RESOURCES,
    clinically_restricted,
    expected_sample_count,
    matrix_urls_for_release,
    resource,
)


def test_every_claim_names_a_paper_location_and_a_code_anchor() -> None:
    assert len(claims()) >= 35
    for claim in claims():
        claim.validate()
        assert claim.paper_location
        assert claim.code_anchors
        assert claim.executability in {
            EXECUTABLE,
            BLOCKED_DATA,
            DESIGN_ONLY,
            NOT_REPORTED,
        }
    assert claim_by_id("eq1-objective").kind == "equation"
    assert claims_of_kind("table")
    with pytest.raises(ConfigError):
        claim_by_id("no-such-claim")


def test_the_anchor_inventory_points_at_modules_that_exist() -> None:
    inventory = anchor_inventory()
    assert inventory
    root = project_root()
    missing: list[str] = []
    for entry in inventory:
        anchor = str(entry["anchor"])
        if anchor.endswith((".py", ".yaml", ".md")):
            if not (root / anchor).exists():
                missing.append(anchor)
            continue
        if not _module_resolves(root, anchor):
            missing.append(anchor)
    assert missing == []


def _module_resolves(root: Path, anchor: str) -> bool:
    """Whether an anchor names an importable module, allowing a class or function suffix.

    An anchor such as ``prism.objectives.total.ConstrainedObjective.stage_two`` names a method
    inside a module, so trailing components are stripped until a module file or a package directory
    is found.
    """
    parts = anchor.split(".")
    while parts:
        candidate = root / "code" / Path(*parts)
        if candidate.with_suffix(".py").is_file() or candidate.is_dir():
            return True
        parts.pop()
    return False


def test_the_context_records_what_the_article_does_not_fix() -> None:
    context = derived_context()
    assert context["field_count"] == len(FIELDS)
    assert isinstance(context["fields"], list)
    assert field_value("equivalence_margin") == pytest.approx(0.015)
    assert field_value("null_band") == pytest.approx(0.003)
    assert field_value("compute_target") == "COMPUTE_NOT_REPORTED"
    assert "hparams_reference" in {
        field["name"] for field in context["fields"] if isinstance(field, dict)
    }
    assert "site_to_region_assignment" in reconstructions()
    assert engineering_defaults() is not None
    assert any(field.origin == ENGINEERING for field in FIELDS)
    assert "engineering_default_values" in engineering_defaults()


def test_the_status_roll_up_never_promotes_an_unexecuted_claim() -> None:
    passed = ClaimOutcome("eq1-objective", PASS, CODE_BEHAVIOUR, "forward-backward", "held")
    failed = ClaimOutcome("table7-scaling", FAIL, MANUSCRIPT_ARITHMETIC, "scaling", "did not hold")
    blocked = ClaimOutcome("table1-predictors", BLOCKED, CODE_BEHAVIOUR, "read", "no access")
    skipped = ClaimOutcome("alg1-early-stop", NOT_RUN, CODE_BEHAVIOUR, "todo", "no execution")
    assert overall_status((passed,)) == VERIFIED
    assert overall_status((blocked, skipped)) == UNVERIFIED
    assert overall_status((passed, failed)) == PARTIALLY_VERIFIED
    assert overall_status(()) == UNVERIFIED
    assert layer_status((passed, failed), MANUSCRIPT_ARITHMETIC) == FAIL
    assert layer_status((passed,), CODE_BEHAVIOUR) == PASS
    assert layer_status((), CODE_BEHAVIOUR) == NOT_RUN
    assert tally((passed, failed, blocked, skipped)) == {
        PASS: 1,
        FAIL: 1,
        BLOCKED: 1,
        NOT_RUN: 1,
    }
    assert failing_claims((passed, failed)) == ("table7-scaling",)
    assert blocked_claims((blocked,)) == ("table1-predictors",)
    with pytest.raises(ConfigError):
        ClaimOutcome("x", "MAYBE", CODE_BEHAVIOUR, "c", "d").validate()
    with pytest.raises(ConfigError):
        require_execution_evidence((blocked, skipped))


def test_the_verification_report_and_claim_map_round_trip(tmp_path: Path) -> None:
    outcomes = (
        ClaimOutcome("eq1-objective", PASS, CODE_BEHAVIOUR, "probe", "held"),
        ClaimOutcome("table7-scaling", FAIL, MANUSCRIPT_ARITHMETIC, "check", "did not hold"),
    )
    report = VerificationReport(
        project="prism", outcomes=outcomes, environment={"python": "3.11"}, notes=("note",)
    )
    target = write_verification_report(tmp_path / "verification_report.json", report)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["overall_status"] == PARTIALLY_VERIFIED
    assert payload["code_status"] == PASS
    assert payload["manuscript_arithmetic_status"] == FAIL
    assert payload["tally"][PASS] == 1
    claim_map = write_claim_map(
        tmp_path / "claim_to_code.json",
        context=derived_context(),
        verification={"eq1-objective": {"status": PASS, "checks": ["probe"]}},
        extra={"deviations": []},
    )
    body = json.loads(claim_map.read_text(encoding="utf-8"))
    assert body["claim_count"] == len(claims())
    assert body["verification"]["eq1-objective"]["status"] == PASS
    with pytest.raises(ConfigError):
        write_claim_map(
            tmp_path / "bad.json",
            context={},
            verification={"not-a-claim": {"status": PASS}},
        )


def test_the_resource_list_carries_only_verified_links() -> None:
    assert len(RESOURCES) == 9
    assert len(ANNOTATED_SERIES) == 6
    urls = matrix_urls_for_release()
    assert all(url.startswith("https://") for url in urls)
    # Six annotated series, two of which were submitted on two platforms; the three open-tier
    # resources are served through APIs and have no series matrix.
    assert len(urls) == 8
    assert all("gdc.cancer.gov" not in url for url in urls)
    assert all(url.endswith("_series_matrix.txt.gz") for url in urls)
    assert resource("TCGA-COAD").n_samples == 461
    assert expected_sample_count("GSE132668") == 37
    assert resource("GSE68204").matrix_files
    for entry in RESOURCES:
        assert entry.license
        assert entry.verified_on
    assert len(clinically_restricted()) == 3


def test_a_missing_resource_is_reported_rather_than_guessed() -> None:
    with pytest.raises(KeyError):
        resource("GSE000000")
    assert len(matrix_urls_for_release()) > 0


def test_the_hygiene_pass_is_clean_on_this_checkout() -> None:
    root = project_root()
    report = run_hygiene(root, root / "code" / "prism")
    assert report.scanned_files > 100
    assert report.module_count > 90
    assert report.markdown_files in {(), ("README.md",)}
    findings = report.by_check()
    assert findings == {}, findings


def test_the_hygiene_pass_finds_a_planted_defect(tmp_path: Path) -> None:
    # The path is assembled from pieces so that this test file does not itself carry a machine
    # specific path, which the same pass would then report.
    planted = "/" + "Users" + "/" + "someone" + "/project/data"
    package = tmp_path / "code" / "prism"
    package.mkdir(parents=True)
    (package / "leaky.py").write_text(
        f'"""A module with a defect."""\n\nDATA_ROOT = "{planted}"\n', encoding="utf-8"
    )
    report = run_hygiene(tmp_path, package)
    findings = report.by_check()
    assert "absolute-path" in findings
    assert not report.clean()
    reported = [finding for finding in report.findings if finding.check == "absolute-path"]
    assert reported
    assert reported[0].path == "code/prism/leaky.py"
    assert isinstance(reported[0], Finding)


def test_the_phrase_scan_ignores_its_own_scanner_and_the_reports(tmp_path: Path) -> None:
    package = tmp_path / "code" / "prism" / "audit"
    package.mkdir(parents=True)
    text = "FORBIDDEN = (" + ", ".join(f'"{phrase}"' for phrase in FORBIDDEN_PHRASES) + ")"
    (package / "integrity.py").write_text(text, encoding="utf-8")
    report = run_hygiene(tmp_path, tmp_path / "code" / "prism")
    assert "generated-content-phrase" not in report.by_check()
    assert tuple(GENERATED_REPORTS)


def test_the_manifest_lists_the_tree_and_not_itself(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "code" / "b.py").write_text("y = 2\n", encoding="utf-8")
    report = HygieneReport()
    target = tmp_path / "integrity_manifest.json"
    written, entries = build_manifest(tmp_path, target, report)
    assert sorted(entry.relative_path for entry in entries) == ["code/a.py", "code/b.py"]
    assert manifest_covers(written, entries)
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["entry_count"] == 2
    assert "integrity_manifest.json" not in {entry.relative_path for entry in entries}
    (tmp_path / "code" / "c.py").write_text("z = 3\n", encoding="utf-8")

    assert not manifest_covers(written, inventory(tmp_path))


def test_every_public_export_resolves() -> None:
    """A package that lists a symbol in ``__all__`` must still be able to import it."""
    packages = (
        "prism",
        "prism.cohort",
        "prism.modelling",
        "prism.objectives",
        "prism.fitting",
        "prism.measurement",
        "prism.statistics",
        "prism.emission",
        "prism.perturbation",
        "prism.assessment",
        "prism.reference",
        "prism.audit",
        "prism.cli",
    )
    for name in packages:
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", ())
        assert exported, name
        for symbol in exported:
            assert hasattr(module, symbol), f"{name} lists {symbol} but does not define it"


def test_modules_are_discoverable_and_imported() -> None:
    root = project_root()
    names = module_names(root / "code" / "prism")
    assert "prism.objectives.total" in names
    referenced = imported_modules(root / "code")
    assert any(name.startswith("prism.") for name in referenced)
    assert isinstance(referenced, set)
    with pytest.raises(ResourceError):
        digest_file(root / "code" / "prism" / "no_such_module.py")
