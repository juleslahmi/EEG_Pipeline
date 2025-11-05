#!/bin/bash
# ======================================================
#  Run a series of EEG model training experiments
#  across models, CV schemes, and hyperparameters.
# ======================================================


VENV_PATH="/home/lahmi/projects/EEG_ML/pipeline"
source "${VENV_PATH}/bin/activate"

PY="${VENV_PATH}/bin/python"
echo "[$(date +%T)] python: $PY"


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"

cd "${ROOT_DIR}"

DATA_ROOT="/home/lahmi/projects/EEG_ML/Data"

DEVICE="cuda"
EPOCHS=10
BATCH_SIZE=64
LR=0.001
WEIGHT_DECAY=0.0
CV_SCHEME="groupkfold"      # or "groupkfold"
CV_N_SPLITS=5        # only used if groupkfold
SEED=42
MODEL=("shallow")
FOLD=("all")
TAG="m=${MODEL}_lr=${LR}_wd=${WEIGHT_DECAY}_bs=${BATCH_SIZE}_ep=${EPOCHS}_${CV_SCHEME}"
OUTDIR="runs/${TAG}"


CACHE_PATH="runs/cache/all_subjects.npz"
if [[ ! -f "$CACHE_PATH" ]]; then
  time "$PY" -m scripts.build_cache --data-root "$DATA_ROOT" --out "$CACHE_PATH"
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

tiùe "$PY" scripts/run_eval.py \
  --dataset-cache "$CACHE_PATH"\
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --cv-scheme "$CV_SCHEME" \
  --cv-n-splits "$CV_N_SPLITS" \
  --cv-fold "$FOLD" \
  --device "$DEVICE" \
  --outdir "$OUTDIR"


echo "All experiments completed."