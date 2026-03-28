#!/usr/bin/env python3
"""
Inspect BIOT decision-making via gradient-based saliency.

Usage example:
  source pipeline/bin/activate
  python src/biot_inspect.py \
    --dataset-cache runs/cache/all_subjects_target=diagnosis_freq=128.npz \
    --checkpoint runs/m=biot_lr=0.001_wd=0.0001_bs=16_ep=50_loso/biot_loso_fold0/best_params.pt \
    --outdir runs/inspect_biot_fold0 

Outputs (written to --outdir):
  - `saliency_mean.npy` : mean absolute gradients (n_chans, n_times)
  - `saliency_std.npy`  : std of abs gradients across samples
  - `channel_importance.npy` : mean importance per channel
  - `time_importance.npy` : mean importance per timepoint
  - `saliency_heatmap.png` : heatmap of channel x time mean saliency
  - `summary.json` : top channels and stats

This script uses input gradients (saliency) computed for the predicted class.
It is model- and dataset-aware (rebuilds BIOT with correct n_chans/n_times).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from src import data_load
from src.build_model import build_model


def load_bundle(cache_path: str):
    bundle = data_load.load_dataset_from_cache(cache_path)
    ds = bundle['dataset']
    patients = bundle.get('patients', None)
    n_chans = int(bundle['n_chans'])
    n_times = int(bundle['n_times'])
    n_classes = len(np.unique(np.asarray(ds.y, dtype=np.int64)))
    return ds, patients, n_chans, n_times, n_classes


def load_checkpoint_to_model(model, checkpoint_path: Path, device: torch.device):
    # load state dict (checkpoint is an OrderedDict of param tensors)
    state = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_saliency_for_batch(model, x_batch: torch.Tensor, device: torch.device):
    x = x_batch.to(device).requires_grad_(True)
    out = model(x)
    if isinstance(out, tuple) or (isinstance(out, list) and len(out) > 1):
        logits = out[0]
    else:
        logits = out

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    saliency = []
    for i in range(x.shape[0]):
        model.zero_grad()
        logit = logits[i, preds[i]]
        logit.backward(retain_graph=True)
        g = x.grad[i].detach().cpu().abs().numpy()  # (n_chans, n_times)
        saliency.append(g)
        x.grad.zero_()
    saliency = np.stack(saliency, axis=0)
    return saliency  # (B, n_chans, n_times)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-cache', type=str, required=True)
    ap.add_argument('--data-root', type=str, default=None,
                    help='Optional: path to raw Data/ to read channel names for mapping')
    ap.add_argument('--zero-eog', action='store_true',
                    help='If set, zero out detected EOG channels in inputs before computing saliency')
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--model', type=str, default='biot')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--n-samples', type=int, default=256,
                    help='Number of random samples to analyze (epochs).')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--outdir', type=str, default='runs/inspect_biot')
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Loading dataset cache...')
    ds, patients, n_chans, n_times, n_classes = load_bundle(args.dataset_cache)
    print(f'Loaded dataset: n_chans={n_chans}, n_times={n_times}, n_classes={n_classes}, n_samples={len(ds)}')

    model_cfg = {'name': args.model, 'n_chans': n_chans, 'n_times': n_times, 'n_classes': n_classes}
    model = build_model(model_cfg)

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print('Loading checkpoint into model...')
    load_checkpoint_to_model(model, Path(args.checkpoint), device)

    eog_indices = []
    if args.zero_eog:
        if not args.data_root:
            raise ValueError('--zero-eog requires --data-root to map channel names')
        import mne
        files = list(Path(args.data_root).rglob('*.set'))
        if not files:
            raise ValueError(f'No .set files found under {args.data_root}')
        raw_epochs = mne.read_epochs_eeglab(str(files[0]), verbose=False)
        ch_names_full = raw_epochs.ch_names
        eog_keywords = ('EOG', 'VEOG', 'HEOG', 'LOC', 'ROC')
        eog_indices = [i for i, ch in enumerate(ch_names_full) if any(k in ch.upper() for k in eog_keywords)]
        print('Detected EOG channel indices (to be zeroed):', eog_indices)

    rng = np.random.RandomState(42)
    all_idx = np.arange(len(ds))
    if args.n_samples >= len(all_idx):
        sel = all_idx
    else:
        sel = rng.choice(all_idx, size=args.n_samples, replace=False)

    batch_size = int(args.batch_size)
    saliency_list = []
    print(f'Computing saliency on {len(sel)} samples (batch_size={batch_size})...')
    for i in range(0, len(sel), batch_size):
        ids = sel[i:i+batch_size]
        xb = torch.stack([ds[j][0].float() for j in ids], dim=0)
        if args.zero_eog and eog_indices:
            for idx in eog_indices:
                if idx < xb.shape[1]:
                    xb[:, idx, :] = 0.0
        s = compute_saliency_for_batch(model, xb, device)
        saliency_list.append(s)

    saliency = np.concatenate(saliency_list, axis=0)  # (N, n_chans, n_times)
    mean_sal = saliency.mean(axis=0)
    std_sal = saliency.std(axis=0)

    channel_imp = mean_sal.mean(axis=1)  # mean over time -> (n_chans,)
    time_imp = mean_sal.mean(axis=0)     # mean over channels -> (n_times,)

    # per-patient aggregation if patients list present
    patient_imp = None
    if patients is not None:
        patients_arr = np.array(patients)
        unique_p = np.unique(patients_arr)
        p_imp = {}
        for p in unique_p:
            sel_idx = np.where(patients_arr == p)[0]
            # intersect with sel
            sel_in_samples = np.intersect1d(sel, sel_idx, assume_unique=True)
            if sel_in_samples.size == 0:
                continue
            pos = np.nonzero(np.isin(sel, sel_in_samples))[0]
            p_imp[p] = saliency[pos].mean(axis=0).tolist()
        patient_imp = p_imp

    # Save arrays
    np.save(outdir / 'saliency_mean.npy', mean_sal)
    np.save(outdir / 'saliency_std.npy', std_sal)
    np.save(outdir / 'channel_importance.npy', channel_imp)
    np.save(outdir / 'time_importance.npy', time_imp)

    # Save summary and top channels
    top_ch_idx = np.argsort(-channel_imp)[:10]
    summary = {
        'n_samples_analyzed': int(len(sel)),
        'n_chans': int(n_chans),
        'n_times': int(n_times),
        'top_channels': [int(i) for i in top_ch_idx.tolist()],
        'top_channel_importances': [float(channel_imp[i]) for i in top_ch_idx.tolist()],
    }
    if patient_imp is not None:
        summary['n_patients_with_data'] = len(patient_imp)

    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    # Plot heatmap of mean saliency
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_sal, aspect='auto', origin='lower')
    plt.colorbar(label='mean |grad|')
    plt.xlabel('time')
    plt.ylabel('channel')
    plt.title('BIOT mean saliency (channel x time)')
    plt.tight_layout()
    plt.savefig(outdir / 'saliency_heatmap.png', dpi=150)
    plt.close()

    # Save patient-level if present
    if patient_imp is not None:
        (outdir / 'patient_importance.json').write_text(json.dumps(patient_imp))

    print('Done. Outputs written to', outdir)


if __name__ == '__main__':
    main()
