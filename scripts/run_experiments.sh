#!/bin/bash
set -euo pipefail

[ -f .env ] && source .env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"

source "${VENV_PATH}/bin/activate"

PY="${VENV_PATH}/bin/python"
echo "[$(date +%T)] python: $PY"


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"

cd "${ROOT_DIR}"

DEVICE="cuda"
EPOCHS=10
BATCH_SIZE=16
LR=0.005
WEIGHT_DECAY=0.0
CV_SCHEME="loso"      # or "groupkfold"
CV_N_SPLITS=5        # only used if groupkfold
SEED=717
MODEL=("tcn")
FOLD=("all")
TAG="m=${MODEL}_lr=${LR}_wd=${WEIGHT_DECAY}_bs=${BATCH_SIZE}_ep=${EPOCHS}_${CV_SCHEME}"
OUTDIR="runs/${TAG}"


TARGET="diagnosis"
CACHE_PATH="runs/cache/all_subjects_target=${TARGET}.npz"

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[Building cache for target=$TARGET]"
  time "$PY" -m scripts.build_cache \
    --data-root "$DATA_ROOT" \
    --out "$CACHE_PATH" \
    --target "$TARGET"
fi

time "$PY" scripts/run_train.py \
  --dataset-cache "$CACHE_PATH"\
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --cv-scheme "$CV_SCHEME" \
  --cv-n-splits "$CV_N_SPLITS" \
  --cv-fold "$FOLD" \
  --device "$DEVICE" \
  --outdir "$OUTDIR" \
  --seed "$SEED"

time "$PY" scripts/run_eval.py \
  --dataset-cache "$CACHE_PATH"\
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --cv-scheme "$CV_SCHEME" \
  --cv-n-splits "$CV_N_SPLITS" \
  --cv-fold "$FOLD" \
  --device "$DEVICE" \
  --outdir "$OUTDIR"


echo "All experiments completed."