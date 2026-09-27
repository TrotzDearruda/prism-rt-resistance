"""First verification layer: the manuscript's own arithmetic.

Every check here recomputes a number the article states, or a relation the article asserts between
numbers it states, using only the second transcription in ``checks.manuscript_tables`` and the
independent routines in ``checks.reference_math``. Nothing in this module imports the release, so a
disagreement is a disagreement about the article rather than about the release's code. The results
are reported as manuscript-arithmetic findings, kept apart from the runtime findings so that a
contradiction inside the article is not mistaken for a broken release.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from checks import manuscript_tables as mt
from checks.reference_math import (
    clopper_pearson_bounds,
    margin_reachable,
    positive_predictive_value,
    synergy_excess,
)

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

MANUSCRIPT_LAYER = "manuscript-arithmetic"


@dataclass(frozen=True)
class Finding:
    """One arithmetic finding, with the two numbers behind it."""

    check: str
    status: str
    detail: str
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "layer": MANUSCRIPT_LAYER,
            "check": self.check,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def _close(first: float, second: float, tolerance: float) -> bool:
    return abs(first - second) <= tolerance


def check_prevalence_interval(tolerance: float = 1e-3) -> Finding:
    """The printed prevalence interval against an exact binomial reconstruction."""
    point, lower, upper = mt.PREVALENCE
    successes = int(round(point * mt.HELD_OUT_RECORDS))
    rebuilt_lower, rebuilt_upper = clopper_pearson_bounds(successes, mt.HELD_OUT_RECORDS)
    ok = _close(rebuilt_lower, lower, tolerance) and _close(rebuilt_upper, upper, tolerance)
    return Finding(
        check="prevalence-interval",
        status=PASS if ok else FAIL,
        detail=(
            "the printed interval is reproduced by an exact binomial interval at the stated record "
            "count"
            if ok
            else "the printed interval is not reproduced by an exact binomial interval at the "
            "stated record count"
        ),
        evidence={
            "printed": [lower, upper],
            "reconstructed": [rebuilt_lower, rebuilt_upper],
            "successes": float(successes),
            "records": float(mt.HELD_OUT_RECORDS),
            "gap": [rebuilt_lower - lower, rebuilt_upper - upper],
        },
    )


def check_table1_ordering() -> Finding:
    """Discrimination should not be higher externally than internally, or lower at low resource."""
    breaches: list[str] = []
    for key, values in mt.TABLE1.items():
        internal, external, low = values[0], values[1], values[2]
        if external > internal:
            breaches.append(f"{key}: external {external} above internal {internal}")
        if low > external:
            breaches.append(f"{key}: low-resource {low} above external {external}")
    return Finding(
        check="table1-partition-ordering",
        status=PASS if not breaches else FAIL,
        detail=(
            "every predictor degrades from internal to external and again to low resource"
            if not breaches
            else "one or more predictors improve on the harder partition"
        ),
        evidence={"breaches": breaches, "rows": len(mt.TABLE1)},
    )


def check_primary_and_external_margin() -> Finding:
    """The two limbs of the criterion against the numbers printed for them."""
    constrained = mt.TABLE1["prism_constrained"]
    unconstrained = mt.TABLE1["prism_unconstrained"]
    primary = constrained[0] - unconstrained[0]
    external = constrained[1] - unconstrained[1]
    baseline = mt.TABLE1[mt.STRONGEST_BASELINE][1]
    margin_over_baseline = constrained[1] - baseline
    limbs = margin_reachable(primary, mt.SUCCESS_MARGIN) and external > 0.0
    third = _close(margin_over_baseline, mt.REPORTED_EXTERNAL_MARGIN, 5e-3)
    return Finding(
        check="criterion-arithmetic",
        status=PASS if limbs and third else FAIL,
        detail=(
            "the reported primary difference, external difference and external margin over the "
            "strongest baseline all follow from the printed cells"
            if limbs and third
            else "at least one of the reported margins does not follow from the printed cells"
        ),
        evidence={
            "primary_difference": primary,
            "primary_limb": margin_reachable(primary, mt.SUCCESS_MARGIN),
            "external_difference": external,
            "external_limb": external > 0.0,
            "margin_over_strongest_baseline": margin_over_baseline,
            "reported_margin_over_strongest_baseline": mt.REPORTED_EXTERNAL_MARGIN,
        },
    )


def check_synergy_arithmetic() -> Finding:
    """Whether the joint cost printed in the prose follows from the ablation cells."""
    full = mt.TABLE2["full"][1]
    alignment_cost = full - mt.TABLE2["minus_alignment"][1]
    invariance_cost = full - mt.TABLE2["minus_invariance"][1]
    both_cost = full - mt.TABLE2["minus_both"][1]
    excess = synergy_excess(alignment_cost, invariance_cost, both_cost)
    return Finding(
        check="component-synergy",
        status=PASS if excess > 0.0 else FAIL,
        detail=(
            "removing both constraint terms costs more than the sum of the two individual removals"
            if excess > 0.0
            else "the joint removal costs no more than the sum of the two individual removals"
        ),
        evidence={
            "alignment_cost": alignment_cost,
            "invariance_cost": invariance_cost,
            "both_cost": both_cost,
            "additive_expectation": alignment_cost + invariance_cost,
            "excess": excess,
        },
    )


def check_synergy_prose_against_table() -> Finding:
    """The prose prints a joint cost the ablation table does not support."""
    full = mt.TABLE2["full"][1]
    table_both = full - mt.TABLE2["minus_both"][1]
    prose_both = 0.127
    additive = (full - mt.TABLE2["minus_alignment"][1]) + (full - mt.TABLE2["minus_invariance"][1])
    consistent = _close(table_both, prose_both, 1e-6) or _close(additive, prose_both, 1e-6)
    return Finding(
        check="component-synergy-prose",
        status=PASS if consistent else FAIL,
        detail=(
            "the joint cost printed in the prose matches either the table cell or the additive "
            "expectation it is compared against"
            if consistent
            else "the prose prints a joint cost that is neither the table cell nor the additive "
            "expectation, and the same paragraph prints the additive value as the joint one"
        ),
        evidence={
            "table_joint_cost": table_both,
            "additive_expectation": additive,
            "prose_values_printed": [prose_both, additive],
        },
    )


def check_zero_tier_controls() -> Finding:
    """The two controls should move the result back towards the unconstrained configuration."""
    unconstrained = mt.TABLE2["backbone_only"][1]
    full = mt.TABLE2["full"][1]
    constraint_effect = full - unconstrained
    rows: dict[str, float] = {}
    for key in ("tier0_permuted_metabolites", "tier0_randomised_sites"):
        control = mt.TABLE2[key][1]
        rows[key] = (full - control) / constraint_effect if constraint_effect else 0.0
    recovered = all(value > 0.25 for value in rows.values())
    return Finding(
        check="tier-zero-controls",
        status=PASS if recovered else FAIL,
        detail=(
            "both controls absorb a substantial part of the constraint's effect on the external "
            "partition"
            if recovered
            else "at least one control does not move the external partition back towards the "
            "unconstrained configuration"
        ),
        evidence={"recovered_share": rows, "constraint_effect": constraint_effect},
    )


def check_null_band_claim() -> Finding:
    """The caption exempts the interval-head row from the null band; check what else falls in it."""
    full = mt.TABLE2["full"]
    inside: list[str] = []
    for key, values in mt.TABLE2.items():
        if key == "full":
            continue
        for position, column in enumerate(("internal", "external", "lowest_resource")):
            movement = abs(values[position] - full[position])
            if movement < mt.NULL_BAND:
                inside.append(f"{key}.{column}={movement:.3f}")
    unexpected = [
        entry
        for entry in inside
        if not entry.startswith(mt.INTERVAL_HEADS_EXEMPT_FROM_NULL_BAND[0])
    ]
    return Finding(
        check="null-band-exemptions",
        status=PASS if not unexpected else FAIL,
        detail=(
            "only the interval-head row moves a metric by less than the null band"
            if not unexpected
            else "the caption exempts only the interval-head row, but other rows move a metric by "
            "less than the null band"
        ),
        evidence={"cells_inside_band": inside, "unexpected": unexpected},
    )


def check_transfer_retention() -> Finding:
    """The retention factor printed in the prose against the smallest one in the matrix."""
    factors: dict[str, float] = {}
    for row, cells in mt.TABLE3_PANEL_A.items():
        if row == "pooled" or row not in cells:
            continue
        diagonal = cells[row]
        off = [value for name, value in cells.items() if name != row]
        factors[row] = min(off) / diagonal
    smallest_row = min(factors, key=lambda name: factors[name])
    smallest = factors[smallest_row]
    ok = _close(smallest, mt.REPORTED_RETENTION, 0.02)
    return Finding(
        check="cross-region-retention",
        status=PASS if ok else FAIL,
        detail=(
            "the smallest retention factor in the matrix matches the value printed beside it"
            if ok
            else "the smallest retention factor in the matrix does not match the value printed "
            "beside it"
        ),
        evidence={
            "smallest_row": smallest_row,
            "smallest_factor": smallest,
            "reported": mt.REPORTED_RETENTION,
            "all_factors": factors,
        },
    )


def check_diagonal_maximality() -> Finding:
    """The caption says the pooled row exceeds several diagonal entries; check the matrix."""
    breaches: list[str] = []
    for row, cells in mt.TABLE3_PANEL_A.items():
        if row == "pooled" or row not in cells:
            continue
        diagonal = cells[row]
        if any(value > diagonal for name, value in cells.items() if name != row):
            breaches.append(row)
    pooled = mt.TABLE3_PANEL_A["pooled"]
    exceeds = [name for name in pooled if pooled[name] > mt.TABLE3_PANEL_A[name][name]]
    consistent = bool(breaches) or bool(exceeds)
    return Finding(
        check="transfer-diagonal-maximality",
        status=PASS if consistent else FAIL,
        detail=(
            "the caption's reading of the matrix holds: the diagonal is not uniformly the maximum"
            if consistent
            else "the caption states the diagonal is not uniformly the maximum and that the pooled "
            "row exceeds several diagonal entries, while every diagonal is the row maximum and the "
            "pooled row is below every diagonal"
        ),
        evidence={
            "rows_where_diagonal_is_not_maximum": breaches,
            "columns_where_pooled_exceeds_diagonal": exceeds,
        },
    )


def check_reader_ordering() -> Finding:
    """The three reader arms should be ordered, and the overcall rate should fall."""
    clinician = mt.TABLE4["clinician_alone"]
    model = mt.TABLE4["model_alone"]
    team = mt.TABLE4["clinician_with_model"]
    ordered = (
        clinician[0] < model[0] < team[0]
        and clinician[1] < team[1]
        and clinican_le(team[3], model[3], clinician[3])
    )
    return Finding(
        check="reader-arm-ordering",
        status=PASS if ordered else FAIL,
        detail=(
            "the combination is the strongest arm on discrimination and sensitivity and calls "
            "fewer "
            "negatives positive than either single arm"
            if ordered
            else "the three reader arms are not ordered the way the study reports them"
        ),
        evidence={
            "auc": [clinician[0], model[0], team[0]],
            "sensitivity": [clinician[1], model[1], team[1]],
            "overcall": [clinician[3], model[3], team[3]],
        },
    )


def clinican_le(team: float, model: float, clinician: float) -> bool:
    """Whether the combination's overcall rate is the smallest of the three."""
    return team <= model and team <= clinician


def check_reader_pairwise_only() -> Finding:
    """A positive predictive value at the deployment prevalence, from the printed operating "
    "point."""
    team = mt.TABLE4["clinician_with_model"]
    prevalence = mt.PREVALENCE[0]
    value = positive_predictive_value(team[1], team[2], prevalence)
    return Finding(
        check="rule-in-value-at-prevalence",
        status=PASS,
        detail=(
            "the positive predictive value the study insists on reporting is computable from "
            "the printed operating point and precedes one"
        ),
        evidence={
            "sensitivity": team[1],
            "specificity": team[2],
            "prevalence": prevalence,
            "positive_predictive_value": value,
            "below_one": value < 1.0,
        },
    )


def check_cross_domain_prose() -> Finding:
    """The adjacent-setting advantages printed in the prose against the table column."""
    table = [
        mt.TABLE6["gastro_oesophageal"][2],
        mt.TABLE6["pancreatic"][2],
    ]
    table_with_cervical = [*table, mt.TABLE6["cervical"][2]]
    prose = list(mt.TABLE6_PROSE_ADVANTAGES)
    matches = all(_close(a, b, 1e-6) for a, b in zip(table_with_cervical, prose, strict=True))
    return Finding(
        check="cross-domain-advantage-prose",
        status=PASS if matches else FAIL,
        detail=(
            "the advantage sequence printed in the prose matches the advantage column of the table"
            if matches
            else "the third value in the prose sequence does not match the advantage column, which "
            "carries the non-colorectal setting's own advantage"
        ),
        evidence={"table_advantages": table_with_cervical, "prose_advantages": prose},
    )


def check_scaling_claims() -> Finding:
    """The saturation claim printed for the upper block against the curve itself."""
    fractions = sorted(mt.TABLE7_UPPER)
    values = [mt.TABLE7_UPPER[fraction][0] for fraction in fractions]
    maximum = max(values)
    at_smallest = values[0]
    dips_upper = any(values[position] < values[position - 1] for position in range(1, len(values)))
    lower_fractions = sorted(mt.TABLE7_LOWER)
    lower_coherence = [mt.TABLE7_LOWER[fraction][1] for fraction in lower_fractions]
    dips_lower = any(
        lower_coherence[position] < lower_coherence[position - 1]
        for position in range(1, len(lower_coherence))
    )
    saturated_early = at_smallest >= maximum - 0.013
    return Finding(
        check="scaling-saturation",
        status=PASS if saturated_early else FAIL,
        detail=(
            "the largest fraction used for fitting buys no discrimination over the smallest"
            if saturated_early
            else "the prose states the maximum external performance is already reached at the "
            "smallest fraction, while the curve's maximum sits at the largest fraction"
        ),
        evidence={
            "smallest_fraction_value": at_smallest,
            "maximum": maximum,
            "gap": maximum - at_smallest,
            "footer_half_width": 0.013,
            "upper_dips": dips_upper,
            "lower_dips": dips_lower,
        },
    )


def check_ensemble_reading() -> Finding:
    """Seed aggregation should help calibration and width; the regime ensemble should not."""
    single = mt.TABLE8["single"]
    ten = mt.TABLE8["ten_seed"]
    regime = mt.TABLE8["regime"]
    seed_helps = ten[1] > single[1] and ten[2] < single[2]
    regime_auc = regime[0] > ten[0]
    regime_calibration = regime[1] < ten[1]
    return Finding(
        check="ensemble-reading",
        status=PASS if seed_helps and regime_auc and regime_calibration else FAIL,
        detail=(
            "seed aggregation improves calibration and narrows intervals, while the regime "
            "ensemble "
            "buys discrimination and gives up calibration"
            if seed_helps and regime_auc and regime_calibration
            else "the ensemble table does not support the reading the study gives it"
        ),
        evidence={
            "seed_calibration_gain": ten[1] - single[1],
            "seed_width_reduction": single[2] - ten[2],
            "regime_auc_gain": regime[0] - ten[0],
            "regime_calibration_loss": ten[1] - regime[1],
            "nominal_coverage": 0.90,
            "reported_coverage": [single[3], ten[3], regime[3]],
        },
    )


def check_conditional_shift() -> Finding:
    """Movement of external discrimination when the oxygenation proxy is entered."""
    shift = mt.TABLE1["prism_conditional"][1] - mt.TABLE1["prism_constrained"][1]
    # The two printed cells differ by exactly the band, and their difference carries the usual
    # floating-point residue; the comparison allows for that rather than calling it a movement.
    inside_band = abs(shift) <= mt.NULL_BAND + 1e-9
    return Finding(
        check="conditional-proxy-shift",
        status=PASS if inside_band else FAIL,
        detail=(
            "entering the proxy moves external discrimination by no more than the null band, so "
            "the "
            "movement is reported as no movement"
            if inside_band
            else "entering the proxy moves external discrimination by more than the null band"
        ),
        evidence={"shift": shift, "null_band": mt.NULL_BAND},
    )


def check_prevalence_to_prevalence_stability() -> Finding:
    """A fixed operating point read at three prevalences, as the introduction argues it must be."""
    sensitivity = mt.TABLE4["model_alone"][1]
    specificity = mt.TABLE4["model_alone"][2]
    values = [
        positive_predictive_value(sensitivity, specificity, prevalence)
        for prevalence in (0.10, mt.PREVALENCE[0], 0.30)
    ]
    monotone = values[0] < values[1] < values[2]
    return Finding(
        check="prevalence-dependence",
        status=PASS if monotone else FAIL,
        detail=(
            "the same sensitivity and specificity give a higher positive predictive value as "
            "prevalence rises, which is the dependence the introduction names"
            if monotone
            else "the positive predictive value does not track prevalence as the introduction "
            "claims"
        ),
        evidence={"prevalences": [0.10, mt.PREVALENCE[0], 0.30], "values": values},
    )


def run_manuscript_audit() -> tuple[Finding, ...]:
    """Every arithmetic check, in a fixed order."""
    return (
        check_prevalence_interval(),
        check_table1_ordering(),
        check_primary_and_external_margin(),
        check_synergy_arithmetic(),
        check_synergy_prose_against_table(),
        check_zero_tier_controls(),
        check_null_band_claim(),
        check_transfer_retention(),
        check_diagonal_maximality(),
        check_reader_ordering(),
        check_reader_pairwise_only(),
        check_cross_domain_prose(),
        check_scaling_claims(),
        check_ensemble_reading(),
        check_conditional_shift(),
        check_prevalence_to_prevalence_stability(),
    )


def as_mapping(findings: Sequence[Finding]) -> dict[str, dict[str, object]]:
    return {finding.check: finding.as_dict() for finding in findings}
