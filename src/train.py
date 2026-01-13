from __future__ import annotations

import json
import time
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from braindecode import EEGClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from src.target import TargetView, infer_classes_from

from sklearn.metrics import accuracy_score

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_run_dir(outdir: str | Path, model, run_name: str | None) -> Path:
    outdir = Path(outdir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = run_name or f"{model.__class__.__name__}_{ts}"
    run_dir = outdir / base
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def fit_one_fold(
    dataset,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    model: torch.nn.Module,
    outdir: str | Path = "runs/",
    train_cfg: dict | None = None,
    device: str = "cuda",
    seed: int = 42,
    run_name: str | None = None,
    fold_id: int | None = None,
    target: str = "diagnosis",
    event_map: dict | None = None
):
    """
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Your PatientEpochsDataset; items return (X_epoch, y_label).
    train_idx, valid_idx : np.ndarray
        Indices for this fold.
    model : torch.nn.Module
        Already built network (use your model_build.build_model).
    outdir : str | Path
        Where to write run artifacts.
    train_cfg : dict
        {
          "lr": float (default 1e-3),
          "weight_decay": float (default 0.0),
          "batch_size": int (default 64),
          "max_epochs": int (default 50),
          "scheduler": "plateau" | "cosine" (default "plateau"),
          # plateau options:
          "plateau_factor": 0.5,
          "plateau_patience": 3,
          "plateau_threshold": 1e-3,
          # early stopping:
          "early_stopping": True,
          "early_patience": 16,
          "early_threshold": 1e-4,
        }
    device : "cuda" | "cpu"
        Training device.
    seed : int
        Random seed.
    run_name : str | None
        Optional directory name suffix.

    Returns
    -------
    dict
        {
          "clf": EEGClassifier,
          "run_dir": str,
          "checkpoint_path": str | None,
          "metrics_path": str,
        }
    """
    _set_seed(seed)

    train_cfg = train_cfg or {}
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    batch_size = int(train_cfg.get("batch_size", 64))
    max_epochs = int(train_cfg.get("max_epochs", 50))
    scheduler_kind = str(train_cfg.get("scheduler", "plateau")).lower()

    plateau_factor = float(train_cfg.get("plateau_factor", 0.5))
    plateau_patience = int(train_cfg.get("plateau_patience", 3))
    plateau_threshold = float(train_cfg.get("plateau_threshold", 1e-3))

    use_early = bool(train_cfg.get("early_stopping", True))
    early_patience = int(train_cfg.get("early_patience", 16))
    early_threshold = float(train_cfg.get("early_threshold", 1e-4))

    # Subsets
    wrapped = TargetView(dataset, target=target, event_map=event_map)

    train_set = Subset(wrapped, train_idx)
    valid_set = Subset(wrapped, valid_idx)

    classes = infer_classes_from(wrapped)

    # Device
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        effective_device = "cuda"
    else:
        effective_device = "cpu"

    # Callbacks
    train_acc_cb = EpochScoring("accuracy", name="train_acc", on_train=True, lower_is_better=False)

    if scheduler_kind in ["plateau", "reduce_on_plateau"]:
        scheduler_cb = LRScheduler(
            policy=ReduceLROnPlateau,
            monitor="valid_loss",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
        )
    elif scheduler_kind in ["cosine", "cosineannealing"]:
        scheduler_cb = LRScheduler(
            policy=CosineAnnealingLR,
            T_max=max_epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_kind}")

    callbacks = [train_acc_cb, scheduler_cb]

    if use_early:
        callbacks.append(
            EarlyStopping(
                monitor="valid_loss",
                patience=early_patience,
                threshold=early_threshold,
                lower_is_better=True,
            )
        )

    run_dir = _make_run_dir(outdir, model, run_name)

    ckpt_cb = Checkpoint(
        monitor="valid_loss_best",
        fn_prefix="best_",
        dirname=run_dir,          # ensure files go into this fold's dir
        load_best=True,           # model will be rolled back to best epoch
        )
    callbacks.append(ckpt_cb)
    print(f"[INFO] Target={target}, classes={classes}, n_classes={len(classes)}")
    clf = EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=effective_device,
        train_split=predefined_split(valid_set),
        callbacks=callbacks,
        classes=classes,
    )

    start_time = time.time()
    # Fit
    _ = clf.fit(train_set)

    # Save history
    end_time = time.time()
    metrics_path = run_dir / "metrics.csv"

    hist_list = clf.history.to_list()              # each item is a dict for one epoch
    hist_df = pd.DataFrame(hist_list)              # make a small DataFrame

    # find best epoch by valid_loss
    i_best = int(hist_df['valid_loss'].astype(float).idxmin())

    best_valid_loss = float(hist_df.loc[i_best, 'valid_loss'])
    best_valid_acc_hist = (
        float(hist_df.loc[i_best, 'valid_acc']) if 'valid_acc' in hist_df.columns else None
    )

    # Recompute accuracy with the model currently loaded at best epoch
    y_valid = np.array([valid_set[i][1] for i in range(len(valid_set))], dtype=int)
    y_pred  = clf.predict(valid_set).astype(int)
    epoch_acc = float(accuracy_score(y_valid, y_pred))


    if isinstance(valid_set, Subset):
        pids = np.array([dataset.patients[i] for i in valid_set.indices])
        labels = np.array([dataset.y[i] for i in valid_set.indices], dtype=int)
    else:
        pids = np.array(dataset.patients)
        labels = np.array(dataset.y, dtype=int)

    patient_preds = {}
    patient_labels = {}
    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        votes = y_pred[idx]
        patient_preds[pid] = int(np.bincount(votes).argmax())
        patient_labels[pid] = int(np.bincount(labels[idx]).argmax()) 

    pp = np.array(list(patient_preds.values()), dtype=int)
    yy = np.array(list(patient_labels.values()), dtype=int)
    patient_acc = float(accuracy_score(yy, pp))

    row = {
        "fold": -1 if fold_id is None else int(fold_id),
        "valid_acc": epoch_acc,           # at best epoch (recomputed)
        "patient_acc": patient_acc,       # at best epoch
        "valid_loss": best_valid_loss,    # best, not last
        "train_time_s": round(end_time - start_time, 1),
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    print(f"[INFO] Saved metrics to {metrics_path}")
    # Save a small config snapshot for reproducibility
    (run_dir / "cfg.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "device": effective_device,
                "train_cfg": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "scheduler": scheduler_kind,
                    "plateau_factor": plateau_factor,
                    "plateau_patience": plateau_patience,
                    "plateau_threshold": plateau_threshold,
                    "early_stopping": use_early,
                    "early_patience": early_patience,
                    "early_threshold": early_threshold,
                },
                "model_class": model.__class__.__name__,
            },
            indent=2,
        )
    )


    return {
        "clf": clf,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
    }

