"""Prism: constrained multimodal adaptation for radiotherapy-response programmes.

Ref: Methods Sec. 6.1 -- the objective couples an outcome term with a spatial-metabolic alignment
term, a cross-setting invariance penalty and a sparsity-and-sign penalty on the programme decoder,
under a pre-declared discrimination floor. The package is laid out so that each of those pieces is
importable on its own:

``prism.cohort``        records, care settings, partitions, prevalence and metabolite supervision
``prism.modelling``     per-modality encoders, fusion, the heads and the assembled model
``prism.objectives``    the four terms of Eq. (1) and the feasibility constraint of Eq. (2)
``prism.fitting``       optimiser, schedule, checkpoints, the mini-batch engine and the two stages
``prism.measurement``   discrimination, calibration, decisions, attribution, transfer and agreement
``prism.statistics``    paired comparison, multiplicity control, equivalence and the fixed family
``prism.emission``      the reported programme, its effects and the intervals around them
``prism.perturbation``  the in-vitro assay design and its pre-declared bands
``prism.assessment``    the analysis plan, the ablation matrix and the table assembly
``prism.reference``     the values transcribed from the article and the resources it cites
``prism.audit``         the claim map, release hygiene and verification bookkeeping
``prism.cli``           the command line entry points
"""

from __future__ import annotations

from prism._version import __release_name__, __version__
from prism.errors import (
    CheckpointError,
    CohortError,
    ConfigError,
    FeasibilityError,
    PrismError,
    ProtocolError,
    ResourceError,
    SupervisionError,
)
from prism.notation import (
    DEVELOPMENT_SITES,
    HOLDOUT_SITES,
    MODALITY_STREAMS,
    PROGRAMME_IDS,
    PROGRAMME_NAMES,
    REGIONS,
    RESIDUAL_RESOURCE_STRATA,
    MetabolicSupervision,
    Observation,
    ProgrammeEmission,
)

__all__ = [
    "DEVELOPMENT_SITES",
    "HOLDOUT_SITES",
    "MODALITY_STREAMS",
    "PROGRAMME_IDS",
    "PROGRAMME_NAMES",
    "REGIONS",
    "RESIDUAL_RESOURCE_STRATA",
    "CheckpointError",
    "CohortError",
    "ConfigError",
    "FeasibilityError",
    "MetabolicSupervision",
    "Observation",
    "PrismError",
    "ProgrammeEmission",
    "ProtocolError",
    "ResourceError",
    "SupervisionError",
    "__release_name__",
    "__version__",
]
