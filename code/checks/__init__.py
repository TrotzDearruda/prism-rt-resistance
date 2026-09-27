"""Independent verification for this release.

``reference_math`` re-implements the quantities the release computes, ``manuscript_tables`` is a
second transcription of the article's values, ``manuscript_audit`` checks the article's own
arithmetic, ``runtime_probes`` executes the release, and ``run_all`` folds both layers onto the
registered claims and writes the reports.
"""

from __future__ import annotations

__all__ = [
    "manuscript_audit",
    "manuscript_tables",
    "reference_math",
    "runtime_probes",
]
