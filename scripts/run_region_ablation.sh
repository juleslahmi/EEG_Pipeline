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
MODEL="tcn"               # model to test
LR=0.005
WD=0.000
BS=16
EPOCHS=10
CV="loso"                 # loso or groupkfold
GKF_SPLITS=5               # only used when CV=groupkfold
RS_FREQ=40             # resampling frequency
DATA_ROOT="${DATA_ROOT:-Data}"

# List of seeds to try per region
SEEDS=(42 8 52 47 64 958 35 354)

# Regions to ablate (see src/channel_groups.py for available regions)
REGIONS=("baseline" "TEMPORAL" "FRONTAL" "CENTRAL" "PARIETAL"
        "FRONTAL CENTRAL TEMPORAL PARIETAL"
        "FRONTAL CENTRAL" "FRONTAL PARIETAL" "FRONTAL TEMPORAL"
        "CENTRAL PARIETAL" "CENTRAL TEMPORAL"
        "TEMPORAL PARIETAL"
        "FRONTAL CENTRAL TEMPORAL"
        "FRONTAL CENTRAL PARIETAL"
        "FRONTAL TEMPORAL PARIETAL"
        "CENTRAL TEMPORAL PARIETAL"
)

# ----- Run experiments -----
echo "[DEBUG] Starting region ablation with ${#REGIONS[@]} regions"
for REGION in "${REGIONS[@]}"; do
  echo ""
  echo "=========================================="
  echo "=== REGION ABLATION: ${REGION} ==="
  echo "=========================================="
  
  # Set DROP_REGION for cache naming and build args
  if [[ "$REGION" == "baseline" ]]; then
    DROP_REGION_ARR=()
    CACHE_SUFFIX=""
  else
    # Split region string into array (allows multiple regions: "FRONTAL CENTRAL")
    read -ra DROP_REGION_ARR <<< "$REGION"
    CACHE_SUFFIX="_drop_$(printf "%s" "${DROP_REGION_ARR[*]}" | tr ' ' '_')"
  fi
  
  CACHE_PATH="runs/cache/all_subjects_target=diagnosis_freq=${RS_FREQ}${CACHE_SUFFIX}.npz"
  
  # Build cache if needed
  if [[ ! -f "$CACHE_PATH" ]]; then
    echo "[$(date +%T)] Building dataset cache: $CACHE_PATH (EOG always dropped)"
    BUILD_ARGS=(--data-root "$DATA_ROOT" --out "$CACHE_PATH" --freq "$RS_FREQ")
    
    if [[ ${#DROP_REGION_ARR[@]} -gt 0 ]]; then
      BUILD_ARGS+=(--drop-region "${DROP_REGION_ARR[@]}")
    fi
    
    time "$PY" -m scripts.build_cache "${BUILD_ARGS[@]}"
  else
    echo "[$(date +%T)] Using existing cache: $CACHE_PATH"
  fi
  
  # Base tag for this region
  if [[ "$REGION" == "baseline" ]]; then
    BASE_TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BS}_ep=${EPOCHS}_${CV}_baseline"
  else
    BASE_TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BS}_ep=${EPOCHS}_${CV}_drop=$(printf "%s" "${DROP_REGION_ARR[*]}" | tr ' ' '_')"
  fi
  
  # Run across seeds
  for SEED in "${SEEDS[@]}"; do
    OUTDIR="runs/freq=${RS_FREQ}/${BASE_TAG}/seed=${SEED}"
    
    echo ""
    echo "[$(date +%T)] === Region: ${REGION}, Seed: ${SEED} -> ${OUTDIR} ==="
    
    # Train (skip if already present)
    if [[ -d "$OUTDIR" && $(ls -A "$OUTDIR" 2>/dev/null | grep -q .) ]]; then
      echo "[$(date +%T)] Found existing run dir $OUTDIR, skipping training"
    else
      echo "[$(date +%T)] Training..."
      
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
        --seed "$SEED"
    fi
    
    # Evaluate
    echo "[$(date +%T)] Evaluating..."
    
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
  
  # Aggregate results for this region across seeds
  echo ""
  echo "[$(date +%T)] Aggregating results for region: ${REGION}"
  # Normalize region name for output filename (replace spaces with underscores)
  REGION_NORMALIZED=$(printf "%s" "${REGION}" | tr ' ' '_')
  time "$PY" -m src.aggregate_seeds \
    --runs-root "runs/freq=${RS_FREQ}" \
    --base-tag "$BASE_TAG" \
    --model "$MODEL" \
    --cv "$CV" \
    --out "runs/freq=${RS_FREQ}/aggregate_${REGION_NORMALIZED}_seeds.json"
  echo "[DEBUG] Completed region: ${REGION}"
done

echo ""
echo "=========================================="
echo "=== ALL REGION ABLATIONS COMPLETE ==="
echo "=========================================="
echo "Results saved to runs/freq=${RS_FREQ}/aggregate_*_seeds.json"

deactivate
