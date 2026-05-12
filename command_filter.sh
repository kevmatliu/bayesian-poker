#!/usr/bin/env bash
# Run preflop + postflop combo filtering on sessions in sessions_filter.txt.
# Uses artifacts/global_priors.json and artifacts/player_thetas.json.
#
# Usage:
#   ./command_filter.sh [PLURIBUS_ROOT] [SESSIONS_FILE] [PLAYER ...] [-- EXTRA PYTHON ARGS]
#
# - First two args: pluribus root (default: pluribus) and sessions list file (default: sessions_filter.txt).
# - Next args (optional): up to six distinct player names (default: MrBlue Bill Pluribus).
# - Use -- before any extra flags for the Python module (e.g. --json-out).
#
# Examples:
#   ./command_filter.sh
#   ./command_filter.sh pluribus sessions_split/sessions_filter.txt
#   ./command_filter.sh pluribus sessions_split/sessions_filter.txt Alice Bob Carol
#   ./command_filter.sh pluribus sessions_split/sessions_filter.txt P1 P2 P3 P4 -- --json-out out.json
set -euo pipefail

ROOT="${1:-pluribus}"
SESSIONS_FILE="${2:-sessions_split/sessions_filter.txt}"

if (($# >= 2)); then shift 2
elif (($# >= 1)); then shift 1
fi

PLAYERS=()
while (($# > 0)) && [[ "${1:-}" != "--" ]] && ((${#PLAYERS[@]} < 6)); do
  PLAYERS+=("$1")
  shift
done
if (($# > 0)) && [[ "${1:-}" == "--" ]]; then
  shift
fi

DEFAULT_PLAYERS=(MrBlue Bill Pluribus)
if ((${#PLAYERS[@]} == 0)); then
  PLAYERS=("${DEFAULT_PLAYERS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec .venv/bin/python -m runners.filter_sessions "$ROOT" \
  --sessions-file "$SESSIONS_FILE" \
  --players "${PLAYERS[@]}" \
  --global-priors artifacts/global_priors.json \
  --player-thetas artifacts/player_thetas.json \
  "$@"
