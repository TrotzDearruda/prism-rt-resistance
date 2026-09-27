"""Dispatch ``python -m prism.cli`` onto the named entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from prism.cli import audit as audit_entry
from prism.cli import emit as emit_entry
from prism.cli import evaluate as evaluate_entry
from prism.cli import report as report_entry
from prism.cli import train as train_entry

COMMANDS = {
    "train": train_entry.main,
    "evaluate": evaluate_entry.main,
    "emit": emit_entry.main,
    "report": report_entry.main,
    "audit": audit_entry.main,
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in COMMANDS:
        available = ", ".join(sorted(COMMANDS))
        print(f"usage: python -m prism.cli <{available}> [options]", file=sys.stderr)
        return 2
    command = arguments.pop(0)
    return COMMANDS[command](arguments)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
