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
OUTDIR="runs"
DEVICE="cuda"
EPOCHS=50
BATCH_SIZE=64
LR=0.001
WEIGHT_DECAY=0.0
CV_SCHEME="loso"      # or "groupkfold"
CV_N_SPLITS=10        # only used if groupkfold
SEED=42

# --- Define models you want to test ---
MODELS=("shallow")

# --- Define folds to run ---
# Use "all" to run every fold, or list specific folds: (0 1 2)
FOLDS=("all")

# --- Loop through models and folds ---
for MODEL in "${MODELS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    echo "======================================================"
    echo "Running model: $MODEL | CV: $CV_SCHEME | Fold: $FOLD"
    echo "======================================================"

    python3 scripts/run_train.py \
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

    echo ""
  done
done

echo "All experiments completed."
