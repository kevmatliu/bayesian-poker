#!/usr/bin/env bash
# Train per-player theta (find_theta.py) on sessions listed in sessions_theta.txt.
# Uses artifacts/global_priors.json. Default output: artifacts/player_thetas.json (see find_theta --out).
# Usage: ./command_theta.sh [PLURIBUS_ROOT] [SESSIONS_FILE] [-- extra python args...]
set -euo pipefail
ROOT="${1:-pluribus}"
SESSIONS_FILE="${2:-sessions_theta.txt}"
if (( $# >= 2 )); then shift 2
elif (( $# >= 1 )); then shift 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec .venv/bin/python find_theta.py "$ROOT" \
  --sessions-file "$SESSIONS_FILE" \
  --players MrBlue Bill Pluribus \
  --global-priors artifacts/global_priors.json \
  --preflop-m-lr 2e-3 \
  --postflop-m-lr 2e-3 \
  --preflop-m-batch-size 0 \
  --postflop-m-batch-size 0 \
  "$@"
