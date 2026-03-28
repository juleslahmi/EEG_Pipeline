# src/evaluate.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, f1_score
from braindecode import EEGClassifier
from skorch.helper import predefined_split

from src.build_model import build_model
from src.tcn_supress_warning import replace_dropout2d

def _patient_level_predictions(full_ds, subset, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote per patient over the subset, return (y_true, y_pred) per patient."""
    # Map subset indices back to patient IDs and labels
    if isinstance(subset, Subset):
        idxs = np.asarray(subset.indices)
    else:
        idxs = np.arange(len(subset))

    pids = np.array([full_ds.patients[i] for i in idxs])
    labels = np.array([full_ds.y[i] for i in idxs], dtype=int)

    per_pid_pred, per_pid_true = {}, {}
    for pid in np.unique(pids):
        sel = np.where(pids == pid)[0]
        # Weighted voting or simple count. Here simple count.
        vote = np.bincount(preds[sel]).argmax()
        per_pid_pred[pid] = int(vote)
        per_pid_true[pid] = int(np.bincount(labels[sel]).argmax())

    # Return arrays
    y_pred = np.array([per_pid_pred[k] for k in per_pid_pred], dtype=int)
    y_true = np.array([per_pid_true[k] for k in per_pid_true], dtype=int)
    return y_true, y_pred


def _patient_level_accuracy(full_ds, subset, preds: np.ndarray) -> tuple[float, float]:
    """Original helper, wrapper around new logic."""
    y_true, y_pred = _patient_level_predictions(full_ds, subset, preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1


def evaluate_fold(
    dataset,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    model_cfg: dict,
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> dict:
    valid_set = Subset(dataset, valid_idx)

    # Rebuild model & wrapper
    net = build_model(model_cfg)
    replace_dropout2d(net)
    
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    clf = EEGClassifier(
        module=net,
        criterion=torch.nn.CrossEntropyLoss,
        device=("cuda" if use_cuda else "cpu"),
        train_split=None,         # no validation split for evaluation
        iterator_valid__shuffle=False,
    )
    # Initialize and load weights
    clf.initialize()
    clf.load_params(f_params=str(checkpoint_path))

    # Epoch-level accuracy
    y_true = np.array([valid_set[i][1] for i in range(len(valid_set))], dtype=int)
    y_pred = clf.predict(valid_set)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_f1 = f1_score(y_true, y_pred, average='macro')

    # Patient-level accuracy
    y_true_pat, y_pred_pat = _patient_level_predictions(dataset, valid_set, y_pred)
    patient_acc = accuracy_score(y_true_pat, y_pred_pat)
    patient_f1 = f1_score(y_true_pat, y_pred_pat, average='macro', zero_division=0)

    return {
        "epoch_acc": float(epoch_acc),
        "epoch_f1": float(epoch_f1),
        "patient_acc": float(patient_acc),
        "patient_f1": float(patient_f1),
        "n_valid_epochs": int(len(valid_set)),
        "n_valid_patients": int(len(set([dataset.patients[i] for i in valid_idx]))),
        # Return arrays for global aggregation (not typically for CSV)
        "y_true_epoch": y_true,
        "y_pred_epoch": y_pred,
        "y_true_patient": y_true_pat,
        "y_pred_patient": y_pred_pat,
    }
