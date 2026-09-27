"""Project context derived from the article.

Ref: the manuscript supplied with this release. Every field below records the value in use, the
location it was read from and how confident that reading is. A field the article does not fix
carries an engineering default and says so, because a value invented silently is
indistinguishable from a value the article reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

MANUSCRIPT = "manuscript"
ENGINEERING = "engineering-default"
RECONSTRUCTED = "reconstructed"


@dataclass(frozen=True)
class ContextField:
    """One derived field with its provenance."""

    name: str
    value: object
    origin: str
    confidence: str
    location: str
    note: str = ""

    def as_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "origin": self.origin,
            "confidence": self.confidence,
            "location": self.location,
            "note": self.note,
        }


PROJECT_NAME = "prism"
PROJECT_TITLE = (
    "A Multimodal Foundation Model Charting Metabolic Reprogramming Underlying Radiotherapy "
    "Resistance in Colorectal Cancer Across Globally Diverse Care Settings"
)

FIELDS: tuple[ContextField, ...] = (
    ContextField(
        name="project_name",
        value=PROJECT_NAME,
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Introduction, paragraph beginning 'We introduce Prism'",
        note=(
            "the article names its own system; the release keeps that name and the repository "
            "directory carries the full title"
        ),
    ),
    ContextField(
        name="domain",
        value="multimodal oncology representation learning for rectal chemoradiotherapy response",
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Title; Introduction; Methods Sec. 6.1",
    ),
    ContextField(
        name="framework",
        value="PyTorch 2.x with plain torch.nn",
        origin=RECONSTRUCTED,
        confidence=MEDIUM,
        location=(
            "Methods Sec. 6.3 describes encoders, pooling, fusion and a gradient reversal layer"
        ),
        note=(
            "the article states no framework; the grader-based adversarial reading and the "
            "convolutional comparison make a tensor-autodiff stack the only viable reading"
        ),
    ),
    ContextField(
        name="venue",
        value="npj Digital Medicine",
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="file naming and the structured Nature-style front matter",
    ),
    ContextField(
        name="primary_datasets",
        value=(
            "nine public resources across the two panels of Table 5, plus the clinical arm the "
            "study names but does not publish"
        ),
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Results Sec. 2.5; Table 5; Data availability",
        note=(
            "the clinical arm is held by the participating institutions; the six annotated series "
            "and the three open-tier resources are read from their public services"
        ),
    ),
    ContextField(
        name="compute_target",
        value="COMPUTE_NOT_REPORTED",
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="no training-hardware statement anywhere in the article",
        note=(
            "the article reports no GPU type, count, memory or wall clock, and no storage figure; "
            "the release therefore reports its own measured footprint rather than an estimate "
            "presented as the study's"
        ),
    ),
    ContextField(
        name="hparams_reference",
        value="none printed",
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Algorithm 1 caption: no experiment numeric value appears in the algorithm",
        note=(
            "the caption states that all hyper-parameters are declared before fitting, and none "
            "are printed; every learning rate, batch size, epoch count and weight in this release "
            "is an engineering default recorded as one in the run header"
        ),
    ),
    ContextField(
        name="supp_path",
        value="none",
        origin=RECONSTRUCTED,
        confidence=HIGH,
        location="no sibling supplementary file accompanies the manuscript file",
        note="the manuscript is a single 28-page file with no linked supplement",
    ),
    ContextField(
        name="evaluation_unit",
        value="patient",
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Results Sec. 2.6; Methods Sec. 6.10",
    ),
    ContextField(
        name="prevalence",
        value=0.194,
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Results Sec. 2.1",
        note="reported with a 95 per cent interval of [18.2, 20.6]",
    ),
    ContextField(
        name="equivalence_margin",
        value=0.015,
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Methods Sec. 6.9; Results Sec. 2.1",
    ),
    ContextField(
        name="null_band",
        value=0.003,
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Methods Sec. 6.10; Table 2 caption",
    ),
    ContextField(
        name="multiplicity_family",
        value=5,
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Methods Sec. 6.10",
        note="the five contrasts themselves are not printed; the list is a reconstruction",
    ),
    ContextField(
        name="interval_nominal_coverage",
        value=0.90,
        origin=MANUSCRIPT,
        confidence=MEDIUM,
        location=(
            "Table 8, empirical-coverage column, read against the interval sentence in Sec. 6.6"
        ),
        note=(
            "the sentence that describes the interval head also prints a coverage word the table "
            "does not use; the table value governs and the discrepancy is recorded"
        ),
    ),
    ContextField(
        name="resource_strata",
        value=("high-resource", "upper-middle-resource", "lower-middle-resource"),
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Results Sec. 2.6",
    ),
    ContextField(
        name="sites",
        value=("A", "B", "C", "D", "E", "F", "G"),
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Results Sec. 2.6",
    ),
    ContextField(
        name="site_to_region_assignment",
        value=(
            ("A", "Region I"),
            ("B", "Region I"),
            ("C", "Region II"),
            ("D", "Region III"),
            ("E", "Region IV"),
            ("F", "Region III"),
            ("G", "Region IV"),
        ),
        origin=RECONSTRUCTED,
        confidence=LOW,
        location="Results Sec. 2.6 names the sites and the regions but prints no assignment",
        note="reconstruction; the stratum of each site follows the modality-count sentence",
    ),
    ContextField(
        name="metabolite_vocabulary",
        value="48 entries in six programme blocks",
        origin=RECONSTRUCTED,
        confidence=LOW,
        location="Methods Sec. 6.1 names a vocabulary of size q; Sec. 6.6 names the block penalty",
        note=(
            "the contents are not printed; the vocabulary is the named intermediates of the six "
            "programmes the article describes, and its size is an engineering default"
        ),
    ),
    ContextField(
        name="programme_order",
        value=("P1", "P2", "P3", "P4", "P5", "P6"),
        origin=MANUSCRIPT,
        confidence=MEDIUM,
        location=(
            "Results Sec. 2.1 names glycolytic, glutaminolytic, lipid-desaturation and the "
            "hypoxia-linked redox programme; Table 7 prints P1-P4"
        ),
        note="the last two programme names are reconstructed from the Introduction's list",
    ),
    ContextField(
        name="cohort_sizes",
        value={"held_out_records": 3847, "silent_records": 1024},
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Results Sec. 2.1",
    ),
    ContextField(
        name="engineering_default_values",
        value=(
            "the two-stage epoch counts, the batch size, the accumulation window, the world size, "
            "both learning rates, the weight decay, the warmup fraction, the gradient clip, the "
            "precision, the early-stopping patience, the ensemble member count, the modality "
            "dropout rate, the three objective weights, the matching temperature, the "
            "discrimination floor, the encoder widths and depths, the vocabulary size and the "
            "metabolite block contents"
        ),
        origin=ENGINEERING,
        confidence=HIGH,
        location=(
            "Algorithm 1 caption states that no numeric value appears in the algorithm; none is "
            "printed anywhere in the article"
        ),
        note="each of these is declared in configs/ and recorded in the run header as a default",
    ),
    ContextField(
        name="excluded_from_execution",
        value=(
            "the clinical arm, the silent exposure sample and the in-vitro line panel are not "
            "redistributable and are not staged here"
        ),
        origin=MANUSCRIPT,
        confidence=HIGH,
        location="Data availability",
    ),
)


def derived_context() -> Mapping[str, object]:
    """The context block written into the claim map."""
    return {
        "project_title": PROJECT_TITLE,
        "project_name": PROJECT_NAME,
        "fields": [field.as_dict() for field in FIELDS],
        "field_count": len(FIELDS),
        "unresolved_decisions": [
            {
                "field": "hparams_reference",
                "candidates": [
                    "engineering defaults recorded per field, as implemented",
                    "a declaration file that a reviewer fills before publication",
                ],
            },
            {
                "field": "compute_target",
                "candidates": [
                    "report the measured footprint of this release",
                    "leave the field absent until the study reports its own hardware",
                ],
            },
        ],
    }


def field_value(name: str) -> object:
    for field in FIELDS:
        if field.name == name:
            return field.value
    raise KeyError(f"no context field is named {name!r}")


def fields_with_origin(origin: str) -> Sequence[ContextField]:
    return tuple(field for field in FIELDS if field.origin == origin)


def engineering_defaults() -> Sequence[str]:
    """Field names whose value is a default rather than a reported quantity."""
    return tuple(field.name for field in FIELDS if field.origin == ENGINEERING)


def reconstructions() -> Sequence[str]:
    """Field names whose value was reconstructed from the article's prose."""
    return tuple(field.name for field in FIELDS if field.origin == RECONSTRUCTED)
