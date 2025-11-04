#!/bin/bash
# ======================================================
#  Run a series of EEG model training experiments
#  across models, CV schemes, and hyperparameters.
# ======================================================


VENV_PATH="/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/pipeline"
source "${VENV_PATH}/bin/activate"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

DATA_ROOT="/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data"

DEVICE="cuda"
EPOCHS=10
BATCH_SIZE=64
LR=0.001
WEIGHT_DECAY=0.0
CV_SCHEME="loso"      # or "groupkfold"
CV_N_SPLITS=5        # only used if groupkfold
SEED=42
MODEL=("shallow")
FOLD=("all")
TAG="m=${MODEL}_lr=${LR}_wd=${WEIGHT_DECAY}_bs=${BATCH_SIZE}_ep=${EPOCHS}_${CV_SCHEME}_k=${CV_N_SPLITS}"
OUTDIR="runs/${TAG}"

CACHE_PATH="runs/cache/all_subjects.npz"
if [[ ! -f "$CACHE_PATH" ]]; then
  python -m scripts.build_cache --data-root "$DATA_ROOT" --out "$CACHE_PATH"
fi

python3 scripts/run_train.py \
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

python3 scripts/run_eval.py \
  --dataset-cache "$CACHE_PATH"\
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --cv-scheme "$CV_SCHEME" \
  --cv-n-splits "$CV_N_SPLITS" \
  --cv-fold "$FOLD" \
  --device "$DEVICE" \
  --outdir "$OUTDIR"


echo "All experiments completed."
