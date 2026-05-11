#!/usr/bin/env bash
# Run preflop + postflop combo filtering on sessions in sessions_filter.txt.
# Uses artifacts/global_priors.json and artifacts/player_thetas.json.
# Usage: ./command_filter.sh [PLURIBUS_ROOT] [SESSIONS_FILE] [-- extra python args...]
set -euo pipefail
ROOT="${1:-pluribus}"
SESSIONS_FILE="${2:-sessions_filter.txt}"
if (( $# >= 2 )); then shift 2
elif (( $# >= 1 )); then shift 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec .venv/bin/python runner.py filter-sessions "$ROOT" \
  --sessions-file "$SESSIONS_FILE" \
  --players MrBlue Bill Pluribus \
  --global-priors artifacts/global_priors.json \
  --player-thetas artifacts/player_thetas.json \
  "$@"
