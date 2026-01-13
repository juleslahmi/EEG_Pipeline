#!/bin/bash
set -euo pipefail

# Load environment variables if present
[ -f .env ] && source .env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"

# Activate virtual environment
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

PY="python" # Assuming python is in path after activation, or specify path if needed

ROOT_DIR="$REPO_ROOT"
export PYTHONPATH="${ROOT_DIR}"
cd "${ROOT_DIR}"

# Configuration
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data}"
MODEL="tcn"  # Change model if desired (e.g. shallow, tcn, eegnet)
EPOCHS=10
BATCH_SIZE=16
LR=0.001
CV_SCHEME="loso"
SEED=42
FREQ=60
TARGET="diagnosis"

# Cache file to use (will speed up loading)
CACHE_PATH="runs/cache/all_subjects_target=${TARGET}_freq=${FREQ}.npz"

# Check if cache exists, otherwise warn or build it
if [[ ! -f "$CACHE_PATH" ]]; then
    echo "Cache file $CACHE_PATH not found. Attempting to build or load from raw will happen in training."
    # Ideally should build it once here if missing to avoid rebuilding 5 times
    # But run_train.py loads from raw if cache arg is NOT passed. 
    # If cache arg IS passed but file missing, it crashes.
    # So we should build it if missing.
    echo "Building cache..."
    "$PY" -m scripts.build_cache --data-root "$DATA_ROOT" --out "$CACHE_PATH" --target "$TARGET" --freq "$FREQ"
fi

# List of events to exclude one by one
# Check actual events in your dataset if different (e.g. 2011, etc.)
EVENTS=(2011 2021 2031 2041 2051)

for EV in "${EVENTS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running training EXCLUDING event $EV"
    echo "----------------------------------------------------------------"
    
    TAG="ablation_excl_${EV}_m=${MODEL}_target=${TARGET}"
    OUTDIR="runs/ablation/${TAG}"
    
    "$PY" scripts/run_train.py \
        --data-root "$DATA_ROOT" \
        --model "$MODEL" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --cv-scheme "$CV_SCHEME" \
        --device "cuda" \
        --outdir "$OUTDIR" \
        --seed "$SEED" \
        --target "$TARGET" \
        --dataset-cache "$CACHE_PATH" \
        --freq "$FREQ" \
        --exclude-events "$EV"
        
    echo "Running evaluation for excluded event $EV"
    "$PY" scripts/run_eval.py \
        --data-root "$DATA_ROOT" \
        --model "$MODEL" \
        --cv-scheme "$CV_SCHEME" \
        --device "cuda" \
        --outdir "$OUTDIR" \
        --dataset-cache "$CACHE_PATH" \
        --exclude-events "$EV"
    
    echo "Finished run for excluded event $EV. Results in $OUTDIR"
done
