#!/usr/bin/env python3
"""
Aggregate confusion matrices across all seeds for a model run.
Concatenates all predictions and computes single global confusion matrix.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def main():
    ap = argparse.ArgumentParser(description='Aggregate confusion matrices across seeds.')
    ap.add_argument('--model-dir', type=str, required=True,
                    help='Model run directory containing seed=X subdirectories')
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return 1

    # Collect all seed directories
    seed_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith('seed=')])
    if not seed_dirs:
        print(f"No seed directories found under {model_dir}")
        return 1

    print(f"Found {len(seed_dirs)} seeds. Aggregating confusion matrices...")

    # Accumulate predictions
    all_y_true_epoch = []
    all_y_pred_epoch = []
    all_y_true_patient = []
    all_y_pred_patient = []

    for seed_dir in seed_dirs:
        epoch_npy = seed_dir / 'confusion_epoch.npy'
        patient_npy = seed_dir / 'confusion_patient.npy'

        epoch_true = seed_dir / 'y_true_epoch.npy'
        epoch_pred = seed_dir / 'y_pred_epoch.npy'
        patient_true = seed_dir / 'y_true_patient.npy'
        patient_pred = seed_dir / 'y_pred_patient.npy'
        
        if epoch_true.exists() and epoch_pred.exists():
            all_y_true_epoch.append(np.load(epoch_true))
            all_y_pred_epoch.append(np.load(epoch_pred))
        if patient_true.exists() and patient_pred.exists():
            all_y_true_patient.append(np.load(patient_true))
            all_y_pred_patient.append(np.load(patient_pred))

    if not all_y_true_epoch:
        print("[WARN] Predictions not found. Trying to reconstruct from summary.json...")
        print("       (Note: This may not be accurate; ensure predictions are saved during evaluation.)")
        return 1

    # Concatenate
    y_true_epoch = np.concatenate(all_y_true_epoch)
    y_pred_epoch = np.concatenate(all_y_pred_epoch)
    y_true_patient = np.concatenate(all_y_true_patient) if all_y_true_patient else np.array([])
    y_pred_patient = np.concatenate(all_y_pred_patient) if all_y_pred_patient else np.array([])

    # Infer class labels
    unique_true = np.unique(np.concatenate([y_true_epoch, y_true_patient]))
    class_ids = sorted(unique_true.tolist())
    # Map class IDs to meaningful names for diagnosis task
    class_map = {0: 'Control', 1: 'Dyslexic'}
    class_names = [class_map.get(c, str(c)) for c in class_ids]

    # Compute confusion matrices
    cm_epoch = confusion_matrix(y_true_epoch, y_pred_epoch, labels=class_ids)
    if len(y_true_patient) > 0:
        cm_patient = confusion_matrix(y_true_patient, y_pred_patient, labels=class_ids)
    else:
        cm_patient = None

    # Helper to plot and save
    def save_cm(cm, title, out_path):
        # Normalize by row (true label) so each row sums to 100%
        with np.errstate(all='ignore'):
            cm_percent = (cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)) * 100
            cm_percent = np.nan_to_num(cm_percent)

        plt.figure(figsize=(7, 6))
        sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, 
                    annot_kws={'size': 12}, cbar_kws={'label': 'Accuracy (%)'})
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.title(title, fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Saved: {out_path}")

    # Save confusion matrices and reports
    out_dir = Path(model_dir)
    save_cm(cm_epoch, f'Aggregated Epoch Confusion (n_seeds={len(seed_dirs)})',
            out_dir / 'confusion_epoch_agg.png')
    if cm_patient is not None:
        save_cm(cm_patient, f'Aggregated Patient Confusion (n_seeds={len(seed_dirs)})',
                out_dir / 'confusion_patient_agg.png')

    # Save raw matrices
    np.save(out_dir / 'confusion_epoch_agg.npy', cm_epoch)
    if cm_patient is not None:
        np.save(out_dir / 'confusion_patient_agg.npy', cm_patient)

    # Save classification reports
    rep_epoch = classification_report(y_true_epoch, y_pred_epoch, labels=class_ids, target_names=class_names, digits=4)
    rep_str = f"Aggregated Epoch Classification Report (n_seeds={len(seed_dirs)})\n\n{rep_epoch}"
    if cm_patient is not None:
        rep_patient = classification_report(y_true_patient, y_pred_patient, labels=class_ids, target_names=class_names, digits=4)
        rep_str += f"\n\nAggregated Patient Classification Report (n_seeds={len(seed_dirs)})\n\n{rep_patient}"
    (out_dir / 'confusion_reports_agg.txt').write_text(rep_str)
    print(f"  Saved: {out_dir / 'confusion_reports_agg.txt'}")

    print("\nDone. Aggregated confusion matrices saved in", out_dir)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)