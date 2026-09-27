"""A second, independent transcription of the values this release is checked against.

These numbers were read off the manuscript again rather than imported from the release, so a
transcription error in either place shows up as a disagreement instead of passing unnoticed. The
subset here is the one the arithmetic audit actually uses.
"""

from __future__ import annotations

TABLE1: dict[str, tuple[float, float, float, float, float, float]] = {
    # internal, external, low-resource, calibration slope, brier, net benefit
    "clinical_nomogram": (0.681, 0.642, 0.618, 0.71, 0.152, 0.061),
    "mri_radiomics": (0.742, 0.689, 0.663, 0.78, 0.141, 0.078),
    "task_specific_mri": (0.798, 0.712, 0.671, 0.82, 0.133, 0.089),
    "late_fusion": (0.826, 0.744, 0.698, 0.86, 0.127, 0.096),
    "linear_probe": (0.831, 0.751, 0.703, 0.88, 0.125, 0.098),
    "full_fine_tune": (0.869, 0.769, 0.714, 0.84, 0.117, 0.107),
    "prism_unconstrained": (0.904, 0.822, 0.781, 0.91, 0.104, 0.124),
    "prism_constrained": (0.898, 0.861, 0.842, 0.94, 0.105, 0.129),
    "prism_conditional": (0.897, 0.858, 0.840, 0.93, 0.106, 0.127),
}

TABLE2: dict[str, tuple[float, float, float]] = {
    # internal, external, lowest-resource
    "full": (0.898, 0.861, 0.842),
    "minus_alignment": (0.894, 0.823, 0.788),
    "minus_invariance": (0.907, 0.792, 0.741),
    "minus_decoder": (0.901, 0.858, 0.838),
    "minus_interval_head": (0.898, 0.860, 0.841),
    "minus_both": (0.897, 0.734, 0.688),
    "tier0_permuted_metabolites": (0.899, 0.803, 0.762),
    "tier0_randomised_sites": (0.906, 0.786, 0.735),
    "backbone_only": (0.869, 0.769, 0.714),
}

TABLE3_PANEL_A: dict[str, dict[str, float]] = {
    "Region I": {"Region I": 0.909, "Region II": 0.871, "Region III": 0.858, "Region IV": 0.806},
    "Region II": {"Region I": 0.864, "Region II": 0.897, "Region III": 0.849, "Region IV": 0.817},
    "Region III": {"Region I": 0.856, "Region II": 0.843, "Region III": 0.889, "Region IV": 0.821},
    "Region IV": {"Region I": 0.828, "Region II": 0.834, "Region III": 0.838, "Region IV": 0.874},
    "pooled": {"Region I": 0.905, "Region II": 0.874, "Region III": 0.862, "Region IV": 0.842},
}

TABLE4: dict[str, tuple[float, float, float, float]] = {
    # auc, sensitivity at fixed specificity, specificity at fixed sensitivity, overcall rate
    "clinician_alone": (0.786, 0.741, 0.712, 0.184),
    "model_alone": (0.861, 0.829, 0.803, 0.141),
    "clinician_with_model": (0.874, 0.848, 0.821, 0.127),
}

TABLE6: dict[str, tuple[float, float, float, float, float]] = {
    # unconstrained, constrained, advantage, probe, metabolic supervision
    "colorectal": (0.822, 0.861, 0.039, 0.831, 0.861),
    "gastro_oesophageal": (0.774, 0.795, 0.021, 0.762, 0.795),
    "pancreatic": (0.741, 0.759, 0.018, 0.729, 0.759),
    "cervical": (0.761, 0.786, 0.025, 0.748, 0.786),
}

TABLE6_PROSE_ADVANTAGES: tuple[float, ...] = (0.021, 0.018, 0.014)

TABLE7_UPPER: dict[float, tuple[float, float, float]] = {
    0.125: (0.812, 0.87, 0.61),
    0.25: (0.841, 0.90, 0.70),
    0.50: (0.852, 0.92, 0.76),
    0.75: (0.849, 0.91, 0.78),
    1.00: (0.861, 0.94, 0.80),
}

TABLE7_LOWER: dict[float, tuple[float, float, float]] = {
    0.10: (0.829, 0.58, 0.64),
    0.20: (0.843, 0.70, 0.74),
    0.35: (0.854, 0.68, 0.86),
    0.50: (0.856, 0.80, 0.93),
}

TABLE8: dict[str, tuple[float, float, float, float]] = {
    # external auc, calibration slope, mean interval width, empirical coverage
    "single": (0.857, 0.90, 0.041, 0.884),
    "five_seed": (0.861, 0.94, 0.033, 0.897),
    "ten_seed": (0.862, 0.95, 0.031, 0.899),
    "regime": (0.873, 0.93, 0.036, 0.891),
}

PREVALENCE = (0.194, 0.182, 0.206)
HELD_OUT_RECORDS = 3847
SILENT_RECORDS = 1024
SUCCESS_MARGIN = 0.015
NULL_BAND = 0.003
STRONGEST_BASELINE = "full_fine_tune"
REPORTED_EXTERNAL_MARGIN = 0.092
REPORTED_RETENTION = 0.92
INTERVAL_HEADS_EXEMPT_FROM_NULL_BAND = ("minus_interval_head",)
