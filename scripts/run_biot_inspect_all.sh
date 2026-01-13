#!/bin/bash
set -euo pipefail

# Run BIOT inspection for every fold run directory and aggregate results.

[ -f .env ] && source .env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"

source "${VENV_PATH}/bin/activate"

PY="${VENV_PATH}/bin/python"
echo "[$(date +%T)] python: $PY"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"
cd "${ROOT_DIR}"

# Defaults (override via env)
RUNDIR_ROOT="${RUNDIR_ROOT:-runs/m=biot_lr=0.001_wd=0.0001_bs=16_ep=50_loso}"
CACHE_PATH="${CACHE_PATH:-runs/cache/all_subjects_target=diagnosis_freq=128.npz}"
INSPECT_ROOT="${INSPECT_ROOT:-runs/inspect_biot}"
DEVICE="${DEVICE:-cuda}"
N_SAMPLES="${N_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$INSPECT_ROOT"

echo "Inspecting BIOT runs under: $RUNDIR_ROOT"

shopt -s nullglob
for run_dir in "$RUNDIR_ROOT"/*; do
  if [[ -d "$run_dir" ]]; then
    run_name=$(basename "$run_dir")
    ckpt="$run_dir/best_params.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "Skipping $run_name: no checkpoint found at $ckpt"
      continue
    fi
    outdir="$INSPECT_ROOT/$run_name"
    echo "\n===> Inspecting $run_name -> $outdir"
    mkdir -p "$outdir"
    "$PY" src/biot_inspect.py \
      --dataset-cache "$CACHE_PATH" \
      --checkpoint "$ckpt" \
      --outdir "$outdir" \
      --device "$DEVICE" \
      --n-samples "$N_SAMPLES" \
      --batch-size "$BATCH_SIZE"
  fi
done

echo "\nAll folds inspected. Aggregating results..."
"$PY" src/aggregate_biot_inspects.py --inspect-root "$INSPECT_ROOT" --out "$INSPECT_ROOT/aggregate_summary.json"

deactivate
