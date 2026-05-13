#!/usr/bin/env bash
# Train per-player theta with marginal Newton (``python -m runners.find_theta --newton``).
# Default output: artifacts/player_thetas_newton.json (see find_theta --out).
# Usage: ./command_theta_newton.sh [PLURIBUS_ROOT] [SESSIONS_FILE] [-- extra python args...]
# Override interpreter: PYTHON=/path/to/python ./command_theta_newton.sh ...
set -euo pipefail
ROOT="${1:-pluribus}"
SESSIONS_FILE="${2:-sessions_split/sessions_theta.txt}"
if (( $# >= 2 )); then shift 2
elif (( $# >= 1 )); then shift 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ -n "${PYTHON:-}" ]]; then
  _PY="$PYTHON"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  _PY="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  _PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  _PY="$(command -v python)"
else
  echo "command_theta_newton.sh: no Python found. Create .venv (python3 -m venv .venv) or set PYTHON." >&2
  exit 127
fi
exec "$_PY" -m runners.find_theta "$ROOT" \
  --sessions-file "$SESSIONS_FILE" \
  --players MrBlue Bill Pluribus \
  --global-priors artifacts/global_priors.json \
  --out artifacts/player_thetas_newton.json \
  --newton \
  --preflop-m-lr 2e-3 \
  --postflop-m-lr 2e-3 \
  --preflop-m-batch-size 0 \
  --postflop-m-batch-size 0 \
  "$@"
