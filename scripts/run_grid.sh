#!/bin/bash
set -euo pipefail

# --- venv ---
[ -f .env ] && source .env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"

source "${VENV_PATH}/bin/activate"

PY="${VENV_PATH}/bin/python"
echo "[$(date +%T)] python: $PY"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"
cd "${ROOT_DIR}"
# --- data & global settings ---
DEVICE="cuda"
SEED=42

# --- grids ---
MODELS=(shallow deep4 eegnet tcn)
LRS=(0.001 0.0005)
WDS=(0.0 0.0001)
BATCHES=(16 64)
EPOCHS_LIST=(10 50)

CACHE_PATH="runs/cache/all_subjects.npz"
CV_SCHEMES=(loso groupkfold)
# try several GroupKFold sizes; ignored for LOSO
GKF_SPLITS_LIST=(5 10)

for MODEL in "${MODELS[@]}"; do
  for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
      for BS in "${BATCHES[@]}"; do
        for EPOCHS in "${EPOCHS_LIST[@]}"; do
          for CV in "${CV_SCHEMES[@]}"; do
            if [[ ! -f "$CACHE_PATH" ]]; then
                time "$PY" -m scripts.build_cache --data-root "$DATA_ROOT" --out "$CACHE_PATH"
            fi
            if [[ "$CV" == "groupkfold" ]]; then
              for GKF in "${GKF_SPLITS_LIST[@]}"; do
                TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BS}_ep=${EPOCHS}_${CV}-k=${GKF}"
                OUTDIR="runs/${TAG}"
                CV_ARGS=(--cv-scheme groupkfold --cv-n-splits "$GKF" --cv-fold all)

                echo "================ TRAIN ${TAG} ================"
                time "$PY" -m scripts.run_train \
                  --dataset-cache "$CACHE_PATH" \
                  --data-root "$DATA_ROOT" \
                  --model "$MODEL" \
                  --lr "$LR" \
                  --weight-decay "$WD" \
                  --batch-size "$BS" \
                  --epochs "$EPOCHS" \
                  "${CV_ARGS[@]}" \
                  --device "$DEVICE" \
                  --outdir "$OUTDIR" \
                  --seed "$SEED"

                echo "================ EVAL  ${TAG} ================"
                time "$PY" -m scripts.run_eval \
                  --dataset-cache "$CACHE_PATH" \
                  --data-root "$DATA_ROOT" \
                  --model "$MODEL" \
                  "${CV_ARGS[@]}" \
                  --device "$DEVICE" \
                  --outdir "$OUTDIR"
                echo
              done
            else
              # LOSO path (no n_splits parameter)
              TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BS}_ep=${EPOCHS}_${CV}"
              OUTDIR="runs/${TAG}"
              CV_ARGS=(--cv-scheme loso --cv-fold all)

              echo "================ TRAIN ${TAG} ================"
              time "$PY" -m scripts.run_train \
                --dataset-cache "$CACHE_PATH" \
                --data-root "$DATA_ROOT" \
                --model "$MODEL" \
                --lr "$LR" \
                --weight-decay "$WD" \
                --batch-size "$BS" \
                --epochs "$EPOCHS" \
                "${CV_ARGS[@]}" \
                --device "$DEVICE" \
                --outdir "$OUTDIR" \
                --seed "$SEED"

              echo "================ EVAL  ${TAG} ================"
              time "$PY" -m scripts.run_eval \
                --dataset-cache "$CACHE_PATH" \
                --data-root "$DATA_ROOT" \
                --model "$MODEL" \
                "${CV_ARGS[@]}" \
                --device "$DEVICE" \
                --outdir "$OUTDIR"
              echo
            fi

          done
        done
      done
    done
  done
done

echo "Sweep finished. Aggregating results…"
time "$PY" -m scripts.aggregate_results --root runs
echo "See: runs/master_summary.csv"

deactivate
