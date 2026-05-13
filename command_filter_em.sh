#!/usr/bin/env bash
# Same as command_filter.sh but uses EM player thetas. Range CSV auto-writes to
# artifacts/filter_sessions_range_history_em.csv (see runners.filter_sessions --range-csv-out).
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
  --player-thetas artifacts/player_thetas_em.json \
  "$@"
