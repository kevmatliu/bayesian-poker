set -euo pipefail
ROOT="${1:-pluribus}"
SESSIONS_TRAIN_FILE="${2:-sessions_train.txt}"

.venv/bin/python -m runners.train "$ROOT" \
    --sessions-file "$SESSIONS_TRAIN_FILE" \
    --postflop-epochs 100 --postflop-l2 0.001 \
    --out artifacts/global_priors.json
