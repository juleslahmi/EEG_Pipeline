# src/evaluate.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score
from braindecode import EEGClassifier
from skorch.helper import predefined_split

from src.build_model import build_model
from src.tcn_supress_warning import replace_dropout2d

def _patient_level_accuracy(full_ds, subset, preds: np.ndarray) -> float:
    """Majority-vote per patient over the subset, return patient-level accuracy."""
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
        vote = np.bincount(preds[sel]).argmax()
        per_pid_pred[pid] = int(vote)
        # all epochs of a patient have same label; majority is safe:
        per_pid_true[pid] = int(np.bincount(labels[sel]).argmax())

    y_pred = np.array([per_pid_pred[k] for k in per_pid_pred], dtype=int)
    y_true = np.array([per_pid_true[k] for k in per_pid_true], dtype=int)
    return accuracy_score(y_true, y_pred)


def evaluate_fold(
    dataset,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    model_cfg: dict,
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> dict:
    """
    Load a model with best checkpoint and evaluate on the valid subset.
    Returns dict with epoch_acc, patient_acc and counts.
    """
    # Build valid subset
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

    # Patient-level accuracy
    patient_acc = _patient_level_accuracy(dataset, valid_set, y_pred)

    return {
        "epoch_acc": float(epoch_acc),
        "patient_acc": float(patient_acc),
        "n_valid_epochs": int(len(valid_set)),
        "n_valid_patients": int(len(set([dataset.patients[i] for i in valid_idx]))),
    }
