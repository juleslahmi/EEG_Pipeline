#!/bin/bash
set -euo pipefail

# Run BIOT inspection (gradient saliency) with sensible defaults.
# Customize by exporting env vars or passing arguments below.

[ -f .env ] && source .env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"

source "${VENV_PATH}/bin/activate"

PY="${VENV_PATH}/bin/python"
echo "[$(date +%T)] python: $PY"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"
cd "${ROOT_DIR}"

# Defaults (override via env or by editing)
CHECKPOINT="${CHECKPOINT:-runs/m=biot_lr=0.001_wd=0.0001_bs=16_ep=50_loso/biot_loso_fold0/best_params.pt}"
CACHE_PATH="${CACHE_PATH:-runs/cache/all_subjects_target=diagnosis_freq=128.npz}"
OUTDIR="${OUTDIR:-runs/inspect_biot_fold0}"
DEVICE="${DEVICE:-cuda}"
N_SAMPLES="${N_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "Running BIOT inspection"
echo "  checkpoint: $CHECKPOINT"
echo "  cache:      $CACHE_PATH"
echo "  outdir:     $OUTDIR"
echo "  device:     $DEVICE"

"$PY" src/biot_inspect.py \
  --dataset-cache "$CACHE_PATH" \
  --checkpoint "$CHECKPOINT" \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  --n-samples "$N_SAMPLES" \
  --batch-size "$BATCH_SIZE"

deactivate
