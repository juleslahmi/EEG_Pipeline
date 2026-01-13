#!/bin/bash
set -uo pipefail

[ -f .env ] && source .env
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/pipeline}"
source "${VENV_PATH}/bin/activate"
PY="${VENV_PATH}/bin/python"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}"
cd "${ROOT_DIR}"

RS_FREQ=8
MODEL="tcn"
LR=0.005
WD=0.000
BS=16
EPOCHS=10
CV="loso"

# Same regions as run_region_ablation.sh
REGIONS=("FRONTAL CENTRAL TEMPORAL PARIETAL" "baseline" "FRONTAL" "CENTRAL" "TEMPORAL" "PARIETAL"
        "FRONTAL CENTRAL" "FRONTAL TEMPORAL" "FRONTAL PARIETAL"
        "CENTRAL TEMPORAL" "CENTRAL PARIETAL"
        "TEMPORAL PARIETAL"
        "FRONTAL CENTRAL TEMPORAL"
        "FRONTAL CENTRAL PARIETAL"
        "FRONTAL TEMPORAL PARIETAL"
        "CENTRAL TEMPORAL PARIETAL"
)

echo "Running aggregation (nested structure)..."

for REGION in "${REGIONS[@]}"; do
  if [[ "$REGION" == "baseline" ]]; then
    BASE_TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BS}_ep=${EPOCHS}_${CV}_baseline"
  else
    read -ra DROP_REGION_ARR <<< "$REGION"
    BASE_TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BS}_ep=${EPOCHS}_${CV}_drop=$(printf "%s" "${DROP_REGION_ARR[*]}" | tr ' ' '_')"
  fi
  
  REGION_NORMALIZED=$(printf "%s" "${REGION}" | tr ' ' '_')
  OUT_JSON="runs/freq=${RS_FREQ}/aggregate_${REGION_NORMALIZED}_seeds.json"
  
  echo "Aggregating ${REGION} -> ${OUT_JSON}"
  
  "$PY" -m src.aggregate_seeds \
    --runs-root "runs/freq=${RS_FREQ}" \
    --base-tag "$BASE_TAG" \
    --model "$MODEL" \
    --cv "$CV" \
    --out "$OUT_JSON" || echo "Failed to aggregate $REGION (maybe no runs found)"
done
