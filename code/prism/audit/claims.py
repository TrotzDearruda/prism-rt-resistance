"""The claim-to-code map.

Ref: every numbered equation, algorithm, table and stated protocol rule of the article. Each entry
names the paper location it comes from, the code anchors that implement or transcribe it, and
whether the claim can be executed on this machine at all. A claim whose data are held by the
participating institutions is recorded as blocked rather than as passing, because nothing here can
observe it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from prism.errors import ConfigError

EQUATION = "equation"
ALGORITHM = "algorithm"
TABLE = "table"
PROTOCOL = "protocol"
SETTING = "setting"
RESOURCE = "resource"
AMBIGUITY = "ambiguity"

EXECUTABLE = "executable"
BLOCKED_DATA = "blocked-private-cohort"
DESIGN_ONLY = "design-only"
NOT_REPORTED = "not-reported-in-article"


@dataclass(frozen=True)
class Claim:
    """One statement from the article and the code that answers to it."""

    claim_id: str
    kind: str
    statement: str
    paper_location: str
    code_anchors: tuple[str, ...]
    executability: str
    note: str = ""

    def validate(self) -> None:
        if not self.claim_id or not self.statement:
            raise ConfigError("a claim needs an identifier and a statement")
        if not self.code_anchors:
            raise ConfigError(f"claim {self.claim_id} names no code anchor")
        if self.executability not in {
            EXECUTABLE,
            BLOCKED_DATA,
            DESIGN_ONLY,
            NOT_REPORTED,
        }:
            raise ConfigError(f"claim {self.claim_id} carries an unknown executability")

    def as_dict(self) -> Mapping[str, object]:
        self.validate()
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "statement": self.statement,
            "paper_location": self.paper_location,
            "code_anchors": list(self.code_anchors),
            "executability": self.executability,
            "note": self.note,
        }


CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="eq1-objective",
        kind=EQUATION,
        statement=(
            "The constrained objective couples the outcome term with the alignment term, the "
            "invariance penalty and the programme penalty under three weights"
        ),
        paper_location="Methods Sec. 6.1, Eq. (1)",
        code_anchors=(
            "prism.objectives.total.ConstrainedObjective.stage_two",
            "prism.objectives.alignment.AlignmentLoss",
            "prism.objectives.invariance.VariancePenalty",
            "prism.objectives.programme.ProgrammePenalty",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="eq2-feasibility",
        kind=EQUATION,
        statement=(
            "The discrimination of the constrained model must stay at or above a pre-declared "
            "floor, and an epoch that would breach the floor is reverted"
        ),
        paper_location="Methods Sec. 6.1, Eq. (2); Algorithm 1 steps 9-11",
        code_anchors=(
            "prism.objectives.feasibility.FeasibilityFloor",
            "prism.objectives.feasibility.evaluate_epoch",
            "prism.fitting.stages.fit",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="alg1-stage1-probe",
        kind=ALGORITHM,
        statement=(
            "Stage 1 holds the encoders fixed and fits the outcome head, the programme decoder and "
            "the interval head on the outcome term and the programme penalty"
        ),
        paper_location="Algorithm 1 steps 1-2",
        code_anchors=(
            "prism.fitting.stages.fit",
            "prism.modelling.assembly.Prism.freeze_encoders",
            "prism.objectives.total.ConstrainedObjective.stage_one",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="alg1-stage2-constrained",
        kind=ALGORITHM,
        statement=(
            "Stage 2 releases the encoders and takes the outcome, alignment and invariance steps "
            "in turn within each epoch"
        ),
        paper_location="Algorithm 1 steps 4-13",
        code_anchors=(
            "prism.fitting.loop.run_epoch",
            "prism.fitting.loop.outcome_step",
            "prism.fitting.loop.alignment_step",
            "prism.fitting.loop.invariance_step",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="alg1-modality-dropout",
        kind=ALGORITHM,
        statement=(
            "A mini-batch is drawn with modality dropout applied per observation, so any "
            "combination of streams a site can acquire is estimable"
        ),
        paper_location="Algorithm 1 step 5; Methods Sec. 6.3",
        code_anchors=(
            "prism.modelling.modality_dropout.draw_dropout_plan",
            "prism.fitting.loop.apply_dropout",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="alg1-early-stop",
        kind=ALGORITHM,
        statement="Early stopping is taken on the development split only",
        paper_location="Algorithm 1 step 12",
        code_anchors=("prism.fitting.stages.development_split", "prism.fitting.stages.fit"),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="alg1-emission",
        kind=ALGORITHM,
        statement=(
            "The programme is the support of the decoder, the signs come from its coefficients, "
            "and "
            "the effects and intervals come from the interval head validated leave-one-site-out"
        ),
        paper_location="Algorithm 1 steps 15-16",
        code_anchors=(
            "prism.emission.programmes.emit_programmes",
            "prism.emission.intervals.leave_one_site_out_folds",
            "prism.modelling.programme_head.select_support",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="align-two-halves",
        kind=PROTOCOL,
        statement=(
            "The alignment term pulls the fused region representation towards the metabolite block "
            "and matches their second-moment structure, because matching alone admits axes with no "
            "metabolic meaning"
        ),
        paper_location="Methods Sec. 6.4",
        code_anchors=(
            "prism.objectives.alignment.AlignmentLoss.matching",
            "prism.objectives.alignment.AlignmentLoss.second_moment",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="invariance-two-readings",
        kind=PROTOCOL,
        statement=(
            "Two readings of the invariance penalty are implemented: a variance penalty on the "
            "between-site covariance and an adversarial penalty through a reversed gradient"
        ),
        paper_location="Methods Sec. 6.5; Table 3 Panel B",
        code_anchors=(
            "prism.objectives.invariance.VariancePenalty",
            "prism.objectives.invariance.AdversarialPenalty",
            "prism.modelling.assembly.reverse_gradient",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="programme-group-penalty",
        kind=PROTOCOL,
        statement=(
            "Sparsity is imposed by a group penalty over metabolically coherent blocks rather than "
            "element by element"
        ),
        paper_location="Methods Sec. 6.6",
        code_anchors=(
            "prism.objectives.programme.GroupPenalty",
            "prism.cohort.lexicon.Lexicon.index_blocks",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="interval-nominal-coverage",
        kind=PROTOCOL,
        statement=(
            "The interval head targets a fixed nominal reliability, exact under no shift and "
            "measured again under shift"
        ),
        paper_location="Methods Sec. 6.6; Table 8",
        code_anchors=(
            "prism.modelling.interval_head.IntervalHead",
            "prism.emission.intervals.shift_reading",
        ),
        executability=EXECUTABLE,
        note=(
            "the nominal level is printed as 0.90 in the table columns; the sentence that "
            "describes "
            "the head also prints a coverage word the table does not use, so the table value "
            "governs"
        ),
    ),
    Claim(
        claim_id="table1-predictors",
        kind=TABLE,
        statement=(
            "Nine predictors are compared across internal, external and low-resource partitions on "
            "discrimination, calibration, Brier score and decision-curve net benefit"
        ),
        paper_location="Results Sec. 2.1; Table 1",
        code_anchors=(
            "prism.assessment.harness.table1_predictors",
            "prism.reference.tables.TABLE1_ROWS",
        ),
        executability=BLOCKED_DATA,
        note="the retrospective and silent arms are held by the participating institutions",
    ),
    Claim(
        claim_id="table2-ablation",
        kind=TABLE,
        statement=(
            "Nine configurations remove or replace one component of the objective, including two "
            "tier-zero controls"
        ),
        paper_location="Results Sec. 2.2; Table 2",
        code_anchors=(
            "prism.assessment.ablation.ABLATIONS",
            "prism.assessment.ablation.permute_metabolite_rows",
            "prism.assessment.ablation.randomise_site_labels",
            "prism.assessment.ablation.control_recovery",
        ),
        executability=BLOCKED_DATA,
    ),
    Claim(
        claim_id="table2-super-additivity",
        kind=TABLE,
        statement=(
            "Removing both constraint terms costs more than the sum of the individual removals, "
            "which is the super-additivity that carries the architectural reading"
        ),
        paper_location="Results Sec. 2.2",
        code_anchors=("prism.statistics.family.synergy_statistic",),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="table3-transfer-matrix",
        kind=TABLE,
        statement=(
            "A model fitted on one region is evaluated on each other region, and the pooled "
            "site-invariant row is reported even where it exceeds a diagonal cell"
        ),
        paper_location="Results Sec. 2.2; Table 3 Panel A",
        code_anchors=(
            "prism.measurement.transfer.TransferMatrix",
            "prism.cohort.partitions.fold_by_region",
        ),
        executability=BLOCKED_DATA,
    ),
    Claim(
        claim_id="table3-attribution-probe",
        kind=TABLE,
        statement=(
            "A probe trained on the same representations predicts site and region identity, and "
            "the "
            "constrained configurations suppress that prediction while keeping discrimination"
        ),
        paper_location="Results Sec. 2.2; Table 3 Panel B",
        code_anchors=(
            "prism.measurement.attribution.attribution_row",
            "prism.measurement.attribution.constraint_reading",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="table4-reader-study",
        kind=TABLE,
        statement=(
            "Three arms are read on the same cases, and the combination beats both single arms "
            "while "
            "the overcall rate falls"
        ),
        paper_location="Results Sec. 2.3; Table 4",
        code_anchors=(
            "prism.measurement.reader.reader_arm_result",
            "prism.measurement.reader.team_benefit",
            "prism.measurement.reader.automation_bias_reading",
            "prism.measurement.reader.inter_reader_table",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="table5-auxiliary-resources",
        kind=RESOURCE,
        statement=(
            "Three open-tier resources carry no response annotation and six annotated series carry "
            "the auxiliary comparison"
        ),
        paper_location="Results Sec. 2.5; Table 5",
        code_anchors=(
            "prism.reference.urls.RESOURCES",
            "prism.cohort.supply.load_series",
            "prism.cohort.supply.query_gdc_project",
        ),
        executability=EXECUTABLE,
        note=(
            "panel A is not evaluable by design; the annotation columns are marked rather than "
            "filled"
        ),
    ),
    Claim(
        claim_id="table6-cross-domain",
        kind=TABLE,
        statement=(
            "The model transfers without task-specific re-parameterisation to two adjacent tumour "
            "types and one non-colorectal setting, and the advantage decays away from the "
            "primary task"
        ),
        paper_location="Results Sec. 3.1; Table 6",
        code_anchors=(
            "prism.measurement.transfer.TransferAdvantage",
            "prism.measurement.transfer.advantage_reading",
            "prism.measurement.transfer.probe_only_row",
        ),
        executability=BLOCKED_DATA,
    ),
    Claim(
        claim_id="table7-scaling",
        kind=TABLE,
        statement=(
            "The fitting-fraction curve saturates early while the supervision-fraction curve is "
            "still rising, and neither curve is monotone to the reported precision"
        ),
        paper_location="Results Sec. 3.2; Table 7",
        code_anchors=(
            "prism.measurement.scaling.saturation_fraction",
            "prism.measurement.scaling.lower_block_saturation",
            "prism.measurement.scaling.turning_points",
            "prism.cohort.partitions.training_frame",
            "prism.cohort.partitions.stratified_supervision_frame",
        ),
        executability=BLOCKED_DATA,
    ),
    Claim(
        claim_id="table7-coherence",
        kind=TABLE,
        statement=(
            "Programme coherence is the agreement between the emitted programme and the metabolite "
            "covariance structure of the held-out supervised region"
        ),
        paper_location="Table 7 note",
        code_anchors=(
            "prism.measurement.coherence.programme_coherence",
            "prism.cohort.supervision.region_covariance",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="table8-ensembling",
        kind=TABLE,
        statement=(
            "Seed aggregation moves discrimination slightly while helping calibration and "
            "narrowing "
            "intervals; the regime ensemble is the auxiliary comparison"
        ),
        paper_location="Results Sec. 3.3; Table 8",
        code_anchors=(
            "prism.fitting.ensemble.build_seed_ensemble",
            "prism.fitting.ensemble.regime_ensemble",
            "prism.fitting.ensemble.choose_configuration",
            "prism.measurement.uncertainty.aggregation_reading",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="table8-coverage",
        kind=TABLE,
        statement="Empirical interval coverage is reported next to the mean interval width",
        paper_location="Table 8; Methods Sec. 6.10",
        code_anchors=(
            "prism.emission.intervals.coverage_of",
            "prism.emission.intervals.pooled_coverage",
            "prism.modelling.interval_head.report_intervals",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="stat-delong",
        kind=PROTOCOL,
        statement=(
            "Differences in discrimination are assessed with the paired DeLong test over the "
            "pre-specified comparison set"
        ),
        paper_location="Methods Sec. 6.10",
        code_anchors=(
            "prism.statistics.delong.paired_delong",
            "prism.statistics.delong.structural_components",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="stat-holm-family",
        kind=PROTOCOL,
        statement=(
            "The Holm correction is applied inside one family of five pre-declared contrasts and "
            "nothing outside it is corrected"
        ),
        paper_location="Methods Sec. 6.10",
        code_anchors=(
            "prism.statistics.holm.holm_step_down",
            "prism.statistics.family.FAMILY",
            "prism.statistics.holm.exploratory_label",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="stat-bootstrap-patients",
        kind=PROTOCOL,
        statement="Confidence intervals come from bootstrapping at the patient level",
        paper_location="Methods Sec. 6.10",
        code_anchors=(
            "prism.statistics.ci.patient_cluster_bootstrap",
            "prism.numerics.boot.patient_blocks",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="stat-null-band",
        kind=PROTOCOL,
        statement="The null band used to read the ablation table is T = 0.003 as a fraction",
        paper_location="Methods Sec. 6.10; Table 2 caption",
        code_anchors=(
            "prism.statistics.nullband.NullBand",
            "prism.statistics.nullband.coverage_failure",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="stat-equivalence-margin",
        kind=PROTOCOL,
        statement=(
            "The primary limb of the success criterion is an equivalence claim against a "
            "pre-declared margin of 0.015 in the area under the curve"
        ),
        paper_location="Methods Sec. 6.9; Results Sec. 2.1",
        code_anchors=(
            "prism.statistics.equivalence.two_one_sided",
            "prism.statistics.equivalence.SuccessCriterion",
            "prism.statistics.equivalence.reachability_check",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="criterion-conjunctive",
        kind=PROTOCOL,
        statement=(
            "The criterion is conjunctive: parity within the margin on the primary partition and "
            "superiority on the external partition"
        ),
        paper_location="Methods Sec. 6.9; Discussion Sec. 4.1",
        code_anchors=(
            "prism.assessment.protocol.criterion_from_measurements",
            "prism.assessment.protocol.adjudicate_plan",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="conditional-oxygenation",
        kind=PROTOCOL,
        statement=(
            "Entering the oxygenation proxy as a covariate leaves three programmes with "
            "incremental "
            "value, removes the hypoxia-linked one, and leaves the rest inconclusive"
        ),
        paper_location="Results Sec. 2.1; Discussion Sec. 4.1",
        code_anchors=(
            "prism.assessment.conditional.conditional_gain",
            "prism.cohort.pathways.conditional_verdicts",
            "prism.cohort.pathways.programme_specificity_holds",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="prevalence-rule-in",
        kind=PROTOCOL,
        statement=(
            "The complete response rate is 19.4 per cent with a 95 per cent interval of "
            "[18.2, 20.6], rule-in statements are reported as positive predictive values"
        ),
        paper_location="Results Sec. 2.1",
        code_anchors=(
            "prism.cohort.prevalence.reported_prevalence_interval",
            "prism.numerics.operating_point.RuleInProfile",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="partition-patient-level",
        kind=PROTOCOL,
        statement=(
            "Partitioning is at the patient level, sites A-E develop and sites F and G hold "
            "out, and "
            "no cross-setting number comes from a random patient split inside the sites"
        ),
        paper_location="Results Sec. 2.6",
        code_anchors=(
            "prism.cohort.partitions.assign_layers",
            "prism.cohort.partitions.check_no_patient_overlap",
            "prism.cohort.partitions.check_site_boundary",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="perturbation-assay",
        kind=PROTOCOL,
        statement=(
            "The in-vitro arm uses a multi-dose schedule with the perturbation present before and "
            "during irradiation, reports clonogenic survival as a sensitisation ratio, and "
            "fixes the "
            "anticipated bands before the model output is reviewed"
        ),
        paper_location="Methods Sec. 6.7",
        code_anchors=(
            "prism.perturbation.design.IrradiationScheme",
            "prism.perturbation.design.protocol_gate",
            "prism.perturbation.clonogenic.read_assay",
        ),
        executability=DESIGN_ONLY,
        note="no assay is run in this release; the design and its guards are implemented",
    ),
    Claim(
        claim_id="perturbation-flux",
        kind=PROTOCOL,
        statement=(
            "The mechanistic secondary output is stable-isotope-resolved flux, read for whether "
            "the "
            "process advances as anticipated"
        ),
        paper_location="Methods Sec. 6.7",
        code_anchors=(
            "prism.perturbation.flux.isotope_fraction",
            "prism.perturbation.flux.direction_agreement",
        ),
        executability=DESIGN_ONLY,
    ),
    Claim(
        claim_id="perturbation-no-line-swap",
        kind=PROTOCOL,
        statement=(
            "A line that fails the programme is counted as a failed prediction rather than "
            "replaced "
            "by a more convenient line"
        ),
        paper_location="Methods Sec. 6.7",
        code_anchors=(
            "prism.perturbation.falsify.count_failures",
            "prism.perturbation.falsify.line_substitution_requested",
        ),
        executability=EXECUTABLE,
    ),
    Claim(
        claim_id="training-hyperparameters-absent",
        kind=SETTING,
        statement=(
            "No experiment numeric value appears in the algorithm; all hyper-parameters are "
            "declared "
            "before any model is fitted"
        ),
        paper_location="Algorithm 1 caption",
        code_anchors=(
            "prism.provenance.config.ORIGIN_ENGINEERING_DEFAULT",
            "configs/experiment/main.yaml",
        ),
        executability=NOT_REPORTED,
        note=(
            "the article prints no learning rate, batch size, epoch count or weight, so every such "
            "value in this release is an engineering default recorded as one"
        ),
    ),
    Claim(
        claim_id="compute-profile-absent",
        kind=SETTING,
        statement="The compute the study used is not reported",
        paper_location="no compute statement in the article",
        code_anchors=("prism.provenance.config.TrainConfig", "README.md"),
        executability=NOT_REPORTED,
    ),
    Claim(
        claim_id="cohort-assembly-section",
        kind=AMBIGUITY,
        statement=(
            "The section that would give the cohort assembly, endpoints and provenance carries a "
            "heading and no text"
        ),
        paper_location="Methods Sec. 6.2",
        code_anchors=(
            "prism.cohort.archive.read_archive",
            "prism.cohort.strata.DEFAULT_REGISTRY",
        ),
        executability=NOT_REPORTED,
        note=(
            "the site-to-region assignment and the response-endpoint definition are reconstructed "
            "from Results Sec. 2.6 and recorded as a reconstruction"
        ),
    ),
    Claim(
        claim_id="reported-vocabulary-absent",
        kind=AMBIGUITY,
        statement="The metabolite vocabulary size and its blocks are not printed",
        paper_location="Methods Sec. 6.1 and Sec. 6.6 name the vocabulary but not its contents",
        code_anchors=(
            "prism.cohort.lexicon.BLOCK_BY_PROGRAMME",
            "prism.cohort.lexicon.default_lexicon",
        ),
        executability=NOT_REPORTED,
        note=(
            "the vocabulary is the named intermediates of the six programmes the article "
            "describes"
        ),
    ),
    Claim(
        claim_id="internal-contradiction-decoder-section",
        kind=AMBIGUITY,
        statement=(
            "The decoder section is cross-referenced by a section number the article does not "
            "contain"
        ),
        paper_location="Methods Sec. 6.4 refers to a section 9.0 for the sparse decoder",
        code_anchors=("prism.objectives.programme.GroupPenalty",),
        executability=NOT_REPORTED,
        note="the reference is read as Methods Sec. 6.6, which is where the decoder is defined",
    ),
)


def claims() -> tuple[Claim, ...]:
    return CLAIMS


def claim_by_id(claim_id: str) -> Claim:
    for claim in CLAIMS:
        if claim.claim_id == claim_id:
            return claim
    raise ConfigError(f"no claim is registered as {claim_id!r}")


def claims_of_kind(kind: str) -> tuple[Claim, ...]:
    return tuple(claim for claim in CLAIMS if claim.kind == kind)


def claims_with_executability(value: str) -> tuple[Claim, ...]:
    return tuple(claim for claim in CLAIMS if claim.executability == value)


def write_claim_map(
    path: Path,
    context: Mapping[str, object],
    verification: Mapping[str, Mapping[str, object]],
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write the claim map next to the per-claim verification outcomes."""
    body: dict[str, object] = {
        "producer": "prism.audit.claims",
        "project_context": dict(context),
        "claim_count": len(CLAIMS),
        "claims": [claim.as_dict() for claim in CLAIMS],
        "verification": {key: dict(value) for key, value in verification.items()},
    }
    if extra:
        body.update({str(key): value for key, value in extra.items()})
    unknown = [key for key in verification if key not in {claim.claim_id for claim in CLAIMS}]
    if unknown:
        raise ConfigError(f"verification outcomes name unregistered claims: {unknown}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def anchor_inventory() -> Sequence[Mapping[str, object]]:
    """Every distinct code anchor named by the map, with the claims that cite it."""
    seen: dict[str, list[str]] = {}
    for claim in CLAIMS:
        for anchor in claim.code_anchors:
            seen.setdefault(anchor, []).append(claim.claim_id)
    return tuple(
        {"anchor": anchor, "claims": sorted(claims)} for anchor, claims in sorted(seen.items())
    )
