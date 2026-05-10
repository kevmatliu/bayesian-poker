#!/usr/bin/env bash
# Usage:
#   ./command.sh [PLURIBUS_ROOT] [SESSIONS_FILE]
# Example SESSIONS_FILE (one session folder name per line):
#   30
#   31
#   32
set -euo pipefail
ROOT="${1:-pluribus}"
SESSIONS_FILE="${2:-sessions.txt}"

.venv/bin/python runner.py session-split "$ROOT" \
  --sessions-file "$SESSIONS_FILE" \
  --players Gogo Pluribus \
  --seed 42
