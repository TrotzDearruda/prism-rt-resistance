#!/usr/bin/env bash
# Read the held-out partitions and write the reported tables.
#
#   bash scripts/launch_eval.sh [experiment] [extra args...]
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m prism.cli evaluate --experiment "${EXPERIMENT}" "$@"
