#!/usr/bin/env bash
# Default entrypoint: same as command_theta_em.sh (EM → artifacts/player_thetas_em.json).
# Usage: ./command_theta.sh [PLURIBUS_ROOT] [SESSIONS_FILE] [-- extra python args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/command_theta_em.sh" "$@"
