#!/usr/bin/env bash
set -euo pipefail

# ========= user knobs =========
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"     # or your existing venv path
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data}"
DEVICE="${DEVICE:-cuda}"

MODEL="${MODEL:-tcn}"                             # shallow | deep4 | eegnet | tcn | hybridnet
LR="${LR:-0.0005}"
WD="${WD:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-10}"
CV_SCHEME="${CV_SCHEME:-loso}"                    # loso | groupkfold
CV_N_SPLITS="${CV_N_SPLITS:-5}"
SEED="${SEED:-42}"

# export dir name (change if you like)
PKG_NAME="${PKG_NAME:-${MODEL}_lr=${LR}_wd=${WD}_bs=${BATCH_SIZE}_ep=${EPOCHS}_${CV_SCHEME}}"
EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/exports/${PKG_NAME}}"

# --- optional HF push ---
PUSH_TO_HF="${PUSH_TO_HF:-false}"                 # set to "true" to upload
HF_REPO="${HF_REPO:-}"                            # e.g. "username/eeg-${PKG_NAME}"
# ============================

# Activate venv
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "No venv at ${VENV_PATH}. Create one first."
  exit 1
fi
source "${VENV_PATH}/bin/activate"
PY="${VENV_PATH}/bin/python"

# Make cache once (adjust cache keying if needed)
mkdir -p "${REPO_ROOT}/runs/cache"
CACHE_PATH="${REPO_ROOT}/runs/cache/all_subjects.npz"
if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[build_cache] -> $CACHE_PATH"
  time "$PY" -m scripts.build_cache --data-root "$DATA_ROOT" --out "$CACHE_PATH"
fi

# Train
TAG="m=${MODEL}_lr=${LR}_wd=${WD}_bs=${BATCH_SIZE}_ep=${EPOCHS}_${CV_SCHEME}"
OUTDIR="${REPO_ROOT}/runs/${TAG}"
CV_ARGS=(--cv-scheme "${CV_SCHEME}" --cv-fold all)
if [[ "${CV_SCHEME}" == "groupkfold" ]]; then
  CV_ARGS=(--cv-scheme groupkfold --cv-n-splits "${CV_N_SPLITS}" --cv-fold all)
fi

echo "=== TRAIN ${TAG} ==="
time "$PY" -m scripts.run_train \
  --dataset-cache "$CACHE_PATH" \
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --lr "$LR" \
  --weight-decay "$WD" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  "${CV_ARGS[@]}" \
  --device "$DEVICE" \
  --outdir "$OUTDIR" \
  --seed "$SEED"

# Evaluate to produce leaderboard + summary.json
echo "=== EVAL  ${TAG} ==="
time "$PY" -m scripts.run_eval \
  --dataset-cache "$CACHE_PATH" \
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  "${CV_ARGS[@]}" \
  --device "$DEVICE" \
  --outdir "$OUTDIR"

# Package best artifacts
echo "=== PACKAGE → ${EXPORT_DIR} ==="
mkdir -p "$EXPORT_DIR"

# Copy the single-run summary & training cfg snapshot if present
cp -f "${OUTDIR}/summary.json" "$EXPORT_DIR/" 2>/dev/null || true
cp -f "${OUTDIR}/cfg.json" "$EXPORT_DIR/" 2>/dev/null || true

# Copy best checkpoint(s). You may have multiple fold dirs under OUTDIR; include them all.
# This grabs any best_*.pt in the run tree.
find "$OUTDIR" -type f -name "best_*.pt" -print -exec cp -f {} "$EXPORT_DIR/" \;

# Save a small loader example (skorch or pure torch)
cat > "${EXPORT_DIR}/load_example.py" <<'PYCODE'
import numpy as np
import torch
from braindecode import EEGClassifier
# from braindecode.models import TCN, ShallowFBCSPNet, Deep4Net, HybridNet  # pick your model class

def build_model(name, n_chans, n_classes, n_times=None):
    from braindecode.models import TCN, ShallowFBCSPNet, Deep4Net, HybridNet
    if name == "tcn":
        return TCN(n_chans, n_outputs=n_classes)
    elif name == "shallow":
        return ShallowFBCSPNet(n_chans, n_classes, input_window_samples=n_times, final_conv_length="auto")
    elif name == "deep4":
        return Deep4Net(n_chans, n_classes, n_times=n_times, final_conv_length="auto")
    elif name == "hybridnet":
        return HybridNet(n_chans, n_outputs=n_classes, n_times=n_times)
    else:
        raise ValueError(name)

def load_skorch_model(model, f_params, device="cuda", classes=None):
    use_cuda = (device=="cuda" and torch.cuda.is_available())
    clf = EEGClassifier(
        module=model,
        device=("cuda" if use_cuda else "cpu"),
        train_split=None,
        classes=(np.array(classes) if classes is not None else None),
    )
    clf.initialize()
    clf.load_params(f_params=f_params)
    return clf

if __name__ == "__main__":
    # Example usage:
    n_chans, n_classes, n_times = 32, 2, 512
    model = build_model("tcn", n_chans, n_classes)
    clf = load_skorch_model(model, "best_params.pt", device="cuda", classes=[0,1])
    # y_pred = clf.predict(dataset_or_X)
    print("Loaded model from best_params.pt")
PYCODE

# Minimal README (HF model card compatible)
cat > "${EXPORT_DIR}/README.md" <<EOF
# ${PKG_NAME}

Braindecode/PyTorch EEG model export.

**Contents**
- \`best_params.pt\` — PyTorch \`state_dict\` for the best epoch (skorch checkpoint)
- \`summary.json\` — LOSO / GroupKFold summary (means/stds)
- \`cfg.json\` — training config snapshot (if available)
- \`load_example.py\` — minimal loading code

**Load (skorch)**
\`\`\`python
from load_example import build_model, load_skorch_model
model = build_model("${MODEL}", n_chans=32, n_classes=2)
clf = load_skorch_model(model, "best_params.pt", device="cuda", classes=[0,1])
\`\`\`

> Note: Braindecode models are standard PyTorch modules; uploading \`.pt\` + config to the Hub works fine.
EOF

# Create a convenient tarball too
tar -C "$(dirname "$EXPORT_DIR")" -czf "${EXPORT_DIR}.tar.gz" "$(basename "$EXPORT_DIR")"
echo "Wrote ${EXPORT_DIR}.tar.gz"

# Optional: push to Hugging Face Hub
if [[ "${PUSH_TO_HF}" == "true" ]]; then
  if [[ -z "${HF_REPO}" ]]; then
    echo "Set HF_REPO (e.g., username/eeg-${PKG_NAME}) to push. Skipping."
  else
    # requires: pip install huggingface_hub; huggingface-cli login (or HF_TOKEN env)
    python - <<PY
from huggingface_hub import HfApi, create_repo, upload_folder
import os
repo_id = os.environ.get("HF_REPO")
private = bool(int(os.environ.get("HF_PRIVATE", "0")))
api = HfApi()
try:
    create_repo(repo_id=repo_id, exist_ok=True, private=private)
except Exception:
    pass
upload_folder(repo_id=repo_id, folder_path="${EXPORT_DIR}", commit_message="Add ${PKG_NAME}")
print("Pushed to https://huggingface.co/" + repo_id)
PY
  fi
fi

echo "✅ Export ready at: ${EXPORT_DIR}"
