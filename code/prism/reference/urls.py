"""Public resources used for pretraining, world knowledge and the auxiliary comparison.

Ref: Results Sec. 2.5 and Table 5. Panel A lists resources that carry no
radiotherapy-response annotation; Panel B lists the annotated series. Accessions, sample
counts and access route below were read back from the source services rather than taken
from the manuscript text, and the count column matches the N column of Table 5 for every
entry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

VERIFIED_ON = "2026-09-21"

PANEL_PRETRAIN = "pretraining-and-world-knowledge"
PANEL_ANNOTATED = "annotated-radiotherapy-response"


@dataclass(frozen=True)
class DatasetResource:
    """One entry of Table 5 with the route that was used to check it."""

    label: str
    accession: str
    panel: str
    n_samples: int
    endpoint: str
    access_route: str
    landing_url: str
    matrix_dir_url: str
    download_url: str
    license: str
    open_access: bool
    assay: str
    matrix_files: tuple[str, ...] = ()
    verified_on: str = VERIFIED_ON

    def evaluable(self) -> bool:
        return self.panel == PANEL_ANNOTATED

    def matrix_urls(self) -> tuple[str, ...]:
        """Concrete series-matrix URLs, empty for a resource that has no series matrix.

        A series submitted on more than one platform has one matrix per platform and no combined
        matrix, so the file names are recorded rather than guessed. A resource served through an
        API rather than a file directory has no matrix at all, and this returns nothing for it
        rather than composing a URL that would not resolve.
        """
        if not self.matrix_dir_url:
            return ()
        names = self.matrix_files or (f"{self.accession}_series_matrix.txt.gz",)
        return tuple(f"{self.matrix_dir_url}{name}" for name in names)


TCGA_COAD = DatasetResource(
    label="TCGA-COAD (GDC open tier)",
    accession="TCGA-COAD",
    panel=PANEL_PRETRAIN,
    n_samples=461,
    endpoint="not evaluable (no response annotation)",
    access_route="GDC REST API projects endpoint, summary.case_count",
    landing_url="https://portal.gdc.cancer.gov/projects/TCGA-COAD",
    matrix_dir_url="",
    download_url="https://api.gdc.cancer.gov/projects/TCGA-COAD",
    license="NIH Genomic Data Sharing, open tier",
    open_access=True,
    assay="whole-exome, RNA-seq, DNA methylation, copy number",
)

TCGA_READ = DatasetResource(
    label="TCGA-READ (GDC open tier)",
    accession="TCGA-READ",
    panel=PANEL_PRETRAIN,
    n_samples=172,
    endpoint="not evaluable (no response annotation)",
    access_route="GDC REST API projects endpoint, summary.case_count",
    landing_url="https://portal.gdc.cancer.gov/projects/TCGA-READ",
    matrix_dir_url="",
    download_url="https://api.gdc.cancer.gov/projects/TCGA-READ",
    license="NIH Genomic Data Sharing, open tier",
    open_access=True,
    assay="whole-exome, RNA-seq, DNA methylation, copy number",
)

CPTAC_COAD = DatasetResource(
    label="CPTAC-COAD proteogenomic cohort",
    accession="PDC000116",
    panel=PANEL_PRETRAIN,
    n_samples=110,
    endpoint="not evaluable (no response annotation)",
    access_route="Proteomic Data Commons study landing page",
    landing_url="https://proteomic.datacommons.cancer.gov/pdc/study/PDC000116",
    matrix_dir_url="",
    download_url="https://proteomic.datacommons.cancer.gov/pdc/study/PDC000116",
    license="CPTAC data use agreement",
    open_access=True,
    assay="proteome, phosphoproteome, RNA-seq",
)

_GEO_LANDING = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
_GEO_MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{accession}/matrix/"


def _geo(
    label: str,
    accession: str,
    n_samples: int,
    endpoint: str,
    assay: str,
    matrix_files: tuple[str, ...] = (),
) -> DatasetResource:
    prefix = accession[:-3]
    directory = _GEO_MATRIX.format(prefix=prefix, accession=accession)
    names = matrix_files or (f"{accession}_series_matrix.txt.gz",)
    return DatasetResource(
        label=label,
        accession=accession,
        panel=PANEL_ANNOTATED,
        n_samples=n_samples,
        endpoint=endpoint,
        access_route="GEO series accession page and series matrix directory",
        landing_url=f"{_GEO_LANDING}{accession}",
        matrix_dir_url=directory,
        download_url=f"{directory}{names[0]}",
        license="GEO public access, submitter-declared",
        open_access=True,
        assay=assay,
        matrix_files=matrix_files,
    )


GSE132668 = _geo(
    "Rectal chemoradiotherapy series 1",
    "GSE132668",
    37,
    "pCR",
    "DNA methylation array",
)
GSE119409 = _geo(
    "Rectal chemoradiotherapy series 2",
    "GSE119409",
    66,
    "response category",
    "expression array",
)
GSE46862 = _geo(
    "Rectal chemoradiotherapy series 3",
    "GSE46862",
    69,
    "multi-class response",
    "expression array",
)
GSE35452 = _geo(
    "Rectal chemoradiotherapy series 4",
    "GSE35452",
    46,
    "sensitivity",
    "expression array",
)
GSE45404 = _geo(
    "Rectal chemoradiotherapy series 5",
    "GSE45404",
    80,
    "resistance category",
    "expression array (two platforms)",
    matrix_files=(
        "GSE45404-GPL4133_series_matrix.txt.gz",
        "GSE45404-GPL570_series_matrix.txt.gz",
    ),
)
GSE68204 = _geo(
    "Rectal chemoradiotherapy series 6",
    "GSE68204",
    125,
    "pCR",
    "expression array (two platforms)",
    matrix_files=(
        "GSE68204-GPL10850_series_matrix.txt.gz",
        "GSE68204-GPL6480_series_matrix.txt.gz",
    ),
)

RESOURCES: tuple[DatasetResource, ...] = (
    TCGA_COAD,
    TCGA_READ,
    CPTAC_COAD,
    GSE132668,
    GSE119409,
    GSE46862,
    GSE35452,
    GSE45404,
    GSE68204,
)

ANNOTATED_SERIES: tuple[DatasetResource, ...] = tuple(
    resource for resource in RESOURCES if resource.evaluable()
)

BY_ACCESSION: Mapping[str, DatasetResource] = {
    resource.accession: resource for resource in RESOURCES
}


def resource(accession: str) -> DatasetResource:
    return BY_ACCESSION[accession]


def panel_members(panel: str) -> tuple[DatasetResource, ...]:
    return tuple(resource for resource in RESOURCES if resource.panel == panel)


def expected_sample_count(accession: str) -> int:
    """The N printed in Table 5 for this accession."""
    return resource(accession).n_samples


def urls_for_release() -> tuple[str, ...]:
    """The public landing pages, in the order Table 5 lists them."""
    return tuple(resource.landing_url for resource in RESOURCES)


def matrix_urls_for_release() -> tuple[str, ...]:
    """Every matrix URL that was read back from the source service."""
    collected: list[str] = []
    for entry in RESOURCES:
        collected.extend(entry.matrix_urls())
    return tuple(collected)


def clinically_restricted() -> tuple[str, ...]:
    """Resources named by the study that are not redirectable to a public service."""
    return (
        "retrospective neoadjuvant (chemo)radiotherapy arm of the participating sites",
        "silent exposure sample (1,024 records) of the participating sites",
        "in-vitro irradiation arm cell-line panel",
    )
