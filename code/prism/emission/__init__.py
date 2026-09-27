"""Emission: the reported programme, its per-programme effect and the interval around it."""

from __future__ import annotations

from prism.emission.intervals import (
    FoldCoverage,
    cohort_site_counts,
    coverage_of,
    fold_coverages,
    interval_arrays_from_columns,
    interval_from_effect,
    leave_one_site_out_folds,
    pooled_coverage,
    programme_interval_table,
    shift_reading,
)
from prism.emission.programmes import (
    ProgrammeStatement,
    effect_table,
    emission_diagnostics,
    emit_programmes,
    programme_ranking,
    restricted_to,
    sign_agreement_of_statements,
    signs_at_threshold,
    statement_table,
    support_summary,
)

__all__ = [
    "FoldCoverage",
    "ProgrammeStatement",
    "cohort_site_counts",
    "coverage_of",
    "effect_table",
    "emission_diagnostics",
    "emit_programmes",
    "fold_coverages",
    "interval_arrays_from_columns",
    "interval_from_effect",
    "leave_one_site_out_folds",
    "pooled_coverage",
    "programme_interval_table",
    "programme_ranking",
    "restricted_to",
    "shift_reading",
    "sign_agreement_of_statements",
    "signs_at_threshold",
    "statement_table",
    "support_summary",
]
