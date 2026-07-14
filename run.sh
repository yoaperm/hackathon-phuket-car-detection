#!/usr/bin/env bash
#
# Thin runner for the RunPod one-shot batch inference job.
# Assumes it runs from inside the repo directory (deployment via rsync/scp).
# Configuration is via environment variables — see .env.example / RUNPOD.md.
#
# Usage:
#   ./run.sh                 # install deps (once) then run infer.py
#   SKIP_INSTALL=true ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

: "${OUTPUT_DIR:=/workspace/output}"
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

if [[ "${SKIP_INSTALL:-false}" != "true" ]]; then
  echo "==> Installing Python dependencies"
  pip install -r requirements.txt
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

echo "==> Running inference"
python3 infer.py "$@"

echo "==> Output tree ($OUTPUT_DIR):"
find "$OUTPUT_DIR" -maxdepth 3 -type f | sort
