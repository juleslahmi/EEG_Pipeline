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
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)

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

    # Keep and auto-reload best weights (by valid_loss)
    ckpt_cb = Checkpoint(monitor="valid_loss_best", fn_prefix="best_", load_best=True)
    callbacks.append(ckpt_cb)

    # Build classifier
    classes = np.array(sorted(set(dataset.y.tolist() if isinstance(dataset.y, np.ndarray) else dataset.y)))
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

    # Run dir
    run_dir = _make_run_dir(outdir, model, run_name)

    # Fit
    _ = clf.fit(train_set)

    # Save history
    metrics_path = run_dir / "metrics.csv"

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

    # Best checkpoint path (created by Checkpoint callback)
    checkpoint_path = None
    for p in run_dir.glob("best_*.pt"):
        checkpoint_path = str(p)  # last one wins; there should typically be one

    return {
        "clf": clf,
        "run_dir": str(run_dir),
        "checkpoint_path": checkpoint_path,
        "metrics_path": str(metrics_path),
    }
