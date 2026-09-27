#!/usr/bin/env bash
# Stage the public resources the auxiliary comparison names.
#
#   bash scripts/prepare_data.sh
#
# The reader fetches each resource from its own service and records what it received; a resource that
# cannot be reached is reported and skipped rather than substituted. The clinical arm of the study is
# not obtainable here and is not attempted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export PRISM_DATA_ROOT="${ROOT}/data"

mkdir -p "${ROOT}/data"

python3 - <<'PY'
"""Stage every public resource, printing one ledger line per accession."""
from __future__ import annotations

import os
from pathlib import Path

from prism.cohort.supply import load_series, query_gdc_project
from prism.reference.urls import ANNOTATED_SERIES, RESOURCES, expected_sample_count

root = Path(os.environ.get("PRISM_DATA_ROOT", "data"))
cache = root / "geo"
cache.mkdir(parents=True, exist_ok=True)

for entry in RESOURCES:
    try:
        if entry.panel == "annotated-radiotherapy-response":
            bundle = load_series(entry.accession, cache, header_only=False)
            observed = bundle.n_samples()
            note = "complete series matrix"
        else:
            observed = int(query_gdc_project(entry.accession)["case_count"])
            note = "project case count"
    except Exception as failure:
        print(f"{entry.accession}\tunavailable\t{type(failure).__name__}")
        continue
    status = "ok" if observed == expected_sample_count(entry.accession) else "count mismatch"
    print(f"{entry.accession}\t{status}\t{observed} of {expected_sample_count(entry.accession)}\t{note}")

missing = [entry.accession for entry in ANNOTATED_SERIES if not (cache / f"{entry.accession}_series_matrix.txt.gz").exists()]
if missing:
    print("not staged: " + ", ".join(missing))
PY
