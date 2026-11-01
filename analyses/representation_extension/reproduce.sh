#!/usr/bin/env bash
# Reproducibility entrypoint for the representation-extension analysis and
# the manuscript numbers it supports. See README_REPRODUCE.md.
#
#   bash analyses/representation_extension/reproduce.sh              verify (default, no model fits)
#   bash analyses/representation_extension/reproduce.sh --spotcheck  refit 4 banked folds and compare
#   bash analyses/representation_extension/reproduce.sh --full       rerun all 49 cells (days; large machine)
set -euo pipefail
EXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXT_DIR"
MODE="${1:-verify}"
# pip unpacks wheels into TMPDIR; on hosts with a small /tmp (this
# project's VM mounts a 512 MB tmpfs there) the install runs out of
# space, so default TMPDIR to a directory beside the venv.
if [ -z "${TMPDIR:-}" ]; then
  mkdir -p "$EXT_DIR/.tmp-reproduce"
  export TMPDIR="$EXT_DIR/.tmp-reproduce"
fi
VENV=".venv-reproduce"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi
"$VENV/bin/pip" install --quiet -r requirements-reproduce.txt
case "$MODE" in
  verify|--verify)
    exec "$VENV/bin/python" scripts/verify_reproduce.py ;;
  --spotcheck)
    exec env RUNNER_N_JOBS=2 "$VENV/bin/python" scripts/spotcheck.py ;;
  --full)
    exec env RUNNER_N_JOBS=2 "$VENV/bin/python" scripts/run_full.py ;;
  *)
    echo "unknown mode: $MODE (expected verify, --spotcheck, or --full)" >&2
    exit 2 ;;
esac
