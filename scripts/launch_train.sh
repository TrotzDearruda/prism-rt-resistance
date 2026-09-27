#!/usr/bin/env bash
# Launch one reported configuration.
#
#   bash scripts/launch_train.sh [experiment] [world_size] [extra args...]
#
# The world size defaults to the value the configuration declares, so passing it here changes only
# the launch geometry and not the effective batch the configuration records.
set -euo pipefail

EXPERIMENT="${1:-main}"
WORLD_SIZE="${2:-0}"
shift || true
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${WORLD_SIZE}" -gt 1 ]] && command -v torchrun >/dev/null 2>&1; then
  exec torchrun --standalone --nproc_per_node="${WORLD_SIZE}" \
    -m prism.cli train --experiment "${EXPERIMENT}" "$@"
fi

exec python3 -m prism.cli train --experiment "${EXPERIMENT}" "$@"
