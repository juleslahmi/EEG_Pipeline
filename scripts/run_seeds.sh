#!/bin/bash
set -euo pipefail

# Run a single model across multiple random seeds to test generalization.
# Writes per-seed runs into `runs/` and aggregates epoch/patient accuracy across seeds.

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
MODEL="biot"               # model to test
LR=0.0005
WD=0.001
BS=64
EPOCHS=10
CV="loso"                 # loso or groupkfold
GKF_SPLITS=5               # only used when CV=groupkfold
RS_FREQ=60         # resampling frequency (must match dataset cache)
# list of seeds to try
SEEDS=(42 8 52 47 64 958 35 354)

DROP_REGION="FRONTAL CENTRAL TEMPORAL"          # e.g., "frontal", "temporal", "central" (see src/channel_groups.py)
DROP_CHANNELS="P1 P2 P3 P4 P5 P6 P9 P10 Pz PO3 PO4 PO7 PO8 POz O1 O2 Oz Iz"

# dataset cache to use (always drops EOG, optionally drops region)
if [[ -n "$DROP_REGION" ]]; then
  CACHE_SUFFIX="_drop_${DROP_REGION}"
else
  CACHE_SUFFIX=""
fi
CACHE_PATH="runs/cache/N170.npz"


if [[ ! -f "$CACHE_PATH" ]]; then
  echo "Building dataset cache: $CACHE_PATH (EOG always dropped)"
  BUILD_ARGS=(--data-root "$DATA_ROOT" --out "$CACHE_PATH" --freq "$RS_FREQ")
  
  if [[ -n "$DROP_REGION" ]]; then
    BUILD_ARGS+=(--drop-region $DROP_REGION)
  fi
  
  if [[ -n "$DROP_CHANNELS" ]]; then
    BUILD_ARGS+=(--drop-channels $DROP_CHANNELS)
  fi
  
  time "$PY" -m scripts.build_cache "${BUILD_ARGS[@]}"
fi


# output base tag (seed will be appended per-run)
BASE_TAG="N170_BIOT"

for SEED in "${SEEDS[@]}"; do
  TAG="${BASE_TAG}/seed=${SEED}"
  OUTDIR="runs/Components/N170_BIOT/seed=${SEED}"

  echo "\n=== Seed ${SEED} -> OUTDIR=${OUTDIR} ==="

  # Train (skip if already present)
  if [[ -d "$OUTDIR" && $(ls -A "$OUTDIR" 2>/dev/null || true) ]]; then
    echo "Found existing run dir $OUTDIR, skipping training"
  else
    echo "Training seed ${SEED}..."
    if [[ "$CV" == "groupkfold" ]]; then
      CV_ARGS=(--cv-scheme groupkfold --cv-n-splits "$GKF_SPLITS" --cv-fold all)
    else
      CV_ARGS=(--cv-scheme loso --cv-fold all)
    fi

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
      --seed "$SEED" \
      --freq "$RS_FREQ"\
      --tmin 0.100\
      --tmax 0.200
  fi

  # Evaluate (will write leaderboard_{model}_{cv}.csv into OUTDIR)
  echo "Evaluating seed ${SEED}..."
  if [[ "$CV" == "groupkfold" ]]; then
    CV_ARGS=(--cv-scheme groupkfold --cv-n-splits "$GKF_SPLITS" --cv-fold all)
  else
    CV_ARGS=(--cv-scheme loso --cv-fold all)
  fi

  time "$PY" -m scripts.run_eval \
    --dataset-cache "$CACHE_PATH" \
    --data-root "$DATA_ROOT" \
    --model "$MODEL" \
    "${CV_ARGS[@]}" \
    --device "$DEVICE" \
    --outdir "$OUTDIR"

done

time "$PY" -m src.aggregate_seeds \
  --runs-root runs/Components \
  --base-tag "$BASE_TAG" \
  --model "$MODEL" \
  --cv "$CV" \
  --out "runs/Components/aggregated_N170_BIOT.json"
deactivate
