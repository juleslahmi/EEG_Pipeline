#!/bin/bash
set -euo pipefail

# Run models for each time window of a tenth of a second (across seeds).
# Writes per-window/per-seed runs into `runs/time_windows/`.

[ -f .env ] && source .env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"

source "${VENV_PATH}/bin/activate"

PY="${VENV_PATH}/bin/python"
echo "[$(date +%T)] python: $PY"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"
cd "${ROOT_DIR}"

# ----- Config (customize) -----
DEVICE="cuda"
MODEL="tcn"               # model to test
LR=0.005
WD=0.000
BS=16
EPOCHS=10
CV="loso"                 # loso or groupkfold
RS_FREQ=60                # resampling frequency
SEEDS=(42 8 52 47 64 958 35 354)          # list of seeds to try

# Generate time windows: 0.0 to 0.9 (start times)
# We use python to generate the sequence to handle floating point arithmetic
WINDOWS=$($PY -c "import numpy as np; print(' '.join([f'{t:.1f}' for t in np.arange(0.0, 1.0, 0.1)]))")

for TMIN in $WINDOWS; do
    # Calculate TMAX = TMIN + 0.1
    TMAX=$($PY -c "print(f'{float($TMIN) + 0.1:.1f}')")
    
    echo -e "\n=================================================="
    echo "=== Time Window: $TMIN - $TMAX s ==="
    echo "=================================================="
    
    for SEED in "${SEEDS[@]}"; do
        OUTDIR="runs/time_windows/tmin=${TMIN}_tmax=${TMAX}/seed=${SEED}"
        
        echo -e "\n--- Seed ${SEED} -> OUTDIR=${OUTDIR} ---"

        # Train (skip if already present)
        if [[ -d "$OUTDIR" && $(ls -A "$OUTDIR" 2>/dev/null || true) ]]; then
            echo "Found existing run dir $OUTDIR, skipping training"
        else
            echo "Training seed ${SEED}..."
            if [[ "$CV" == "groupkfold" ]]; then
                CV_ARGS=(--cv-scheme groupkfold --cv-n-splits 5 --cv-fold all)
            else
                CV_ARGS=(--cv-scheme loso --cv-fold all)
            fi

            # Note: Not using dataset-cache because it depends on tmin/tmax
            time "$PY" -m scripts.run_train \
                --data-root "$DATA_ROOT" \
                --model "$MODEL" \
                --lr "$LR" \
                --weight-decay "$WD" \
                --batch-size "$BS" \
                --epochs "$EPOCHS" \
                "${CV_ARGS[@]}" \
                --device "$DEVICE" \
                --outdir "$OUTDIR" \
                --seed "$SEED" \
                --freq "$RS_FREQ" \
                --tmin "$TMIN" \
                --tmax "$TMAX"
        fi

        # Evaluate
        echo "Evaluating seed ${SEED}..."
        if [[ "$CV" == "groupkfold" ]]; then
            CV_ARGS=(--cv-scheme groupkfold --cv-n-splits 5 --cv-fold all)
        else
            CV_ARGS=(--cv-scheme loso --cv-fold all)
        fi

        time "$PY" -m scripts.run_eval \
            --data-root "$DATA_ROOT" \
            --model "$MODEL" \
            "${CV_ARGS[@]}" \
            --device "$DEVICE" \
            --outdir "$OUTDIR" \
            --tmin "$TMIN" \
            --tmax "$TMAX"

    done

    # Aggregate seeds for this time window
    echo "Aggregating seeds for time window $TMIN - $TMAX..."
    time "$PY" -m src.aggregate_seeds \
        --runs-root "runs/time_windows" \
        --base-tag "tmin=${TMIN}_tmax=${TMAX}" \
        --model "$MODEL" \
        --cv "$CV" \
        --out "runs/time_windows/summary_tmin=${TMIN}_tmax=${TMAX}.json"
done

echo "All time windows processed."
