from braindecode import EEGClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, EpochScoring
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# src/train.py  (template)
"""
Train one fold: subset the dataset, build model, fit with Skorch/Braindecode,
save metrics and best checkpoint, and return a small RunResult.
"""

def train_one_fold(dataset, fold_spec, model_cfg, train_cfg, outdir, seed):
    """
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Must support indexing; items return (X_epoch, y_label).
        Must expose `y` and `groups` for metadata if you want summaries.
    fold_spec : dict
        {"fold": int, "train_idx": np.ndarray, "valid_idx": np.ndarray, "valid_subjects": list[str]}
    model_cfg : dict
        See src/models.build_model.
    train_cfg : dict
        Expected keys (examples):
          - lr: float
          - weight_decay: float
          - batch_size: int
          - max_epochs: int
          - scheduler: {"type": "reduce_on_plateau" | "cosine", ...}
          - early_stopping: {"patience": int, "monitor": "valid_loss", ...} (optional)
    outdir : str | Path
        Base output directory (e.g., "runs/").
    seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        {
          "run_dir": str,
          "checkpoint_path": str,
          "metrics_path": str,
          "final_summary_path": str,
          # Optionally: "best_valid_acc": float, "patient_acc": float
        }
    """
    # 1) set seeds and create run_dir = outdir / <model_name>/cv=<...>/fold-XX/<timestamp>
    # 2) build train/valid subsets from indices in fold_spec
    # 3) construct model = models.build_model(model_cfg)
    #    - move to CUDA if available / per a flag
    # 4) wrap in EEGClassifier with criterion, optimizer, callbacks:
    #    - CrossEntropyLoss
    #    - AdamW(lr, weight_decay)
    #    - callbacks: EpochScoring('accuracy', on_train=True), LRScheduler(...),
    #                 EarlyStopping(...), Checkpoint(load_best=True), etc.
    # 5) fit: clf.fit(train_set)
    # 6) evaluate:
    #    - epoch-level accuracy on valid_set
    #    - patient-level accuracy by aggregating preds per patient (majority vote)
    # 7) write artifacts in run_dir:
    #    - best.pt (checkpoint)
    #    - metrics.csv (history)
    #    - final.json or final.csv (fold metrics + params snapshot)
    # 8) return RunResult with paths
    pass

clf = EEGClassifier(
    module=model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=1e-3,
    optimizer__weight_decay=0,
    batch_size=64,
    max_epochs=50,
    callbacks=[EpochScoring('accuracy', name='train_acc', on_train=True, lower_is_better=False),
        # Reactive LR: halves LR if valid_loss plateaus
        LRScheduler(policy=ReduceLROnPlateau,
                    factor=0.5, patience=3, threshold=1e-3,
                    monitor='valid_loss'),
        # Stop if no valid_loss improvement
        EarlyStopping(monitor='valid_loss', patience=16, threshold=1e-4,
                      lower_is_better=True),
        # Keep the best epoch and reload it at the end
        Checkpoint(monitor='valid_loss_best', fn_prefix='best_', load_best=True),
    ],
    device="cuda",
    train_split=predefined_split(valid_set),
    classes=np.array([0, 1], dtype=int),
)
