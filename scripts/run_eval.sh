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
EPOCHS=50
BATCH_SIZE=16
LR=0.001
WEIGHT_DECAY=0.0001
CV_SCHEME="loso"      # or "groupkfold"
CV_N_SPLITS=5        # only used if groupkfold
SEED=42
MODEL=("biot")  # shallow, deep4, eegnet, tcn, biot
FOLD=("all")
TAG="m=${MODEL}_lr=${LR}_wd=${WEIGHT_DECAY}_bs=${BATCH_SIZE}_ep=${EPOCHS}_${CV_SCHEME}"
OUTDIR="runs/${TAG}"


TARGET="diagnosis"  # or "diagnosis"
CACHE_PATH="runs/cache/all_subjects_target=${TARGET}_freq=128.npz"
EVENT_MAP='{"2011":0,"2021":1,"2031":2,"2041":3,"2051":4}'

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[Building cache for target=$TARGET]"
  time "$PY" -m scripts.build_cache \
    --data-root "$DATA_ROOT" \
    --out "$CACHE_PATH" \
    --target "$TARGET"\
    --event-map "$EVENT_MAP"
fi

time "$PY" scripts/run_eval.py \
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --cv-scheme "$CV_SCHEME" \
  --cv-n-splits "$CV_N_SPLITS" \
  --cv-fold "$FOLD" \
  --device "$DEVICE" \
  --outdir "$OUTDIR"\
  --dataset-cache "$CACHE_PATH"\