from pathlib import Path
import numpy as np
import torch
from braindecode import EEGClassifier

def _find_fold_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    folds = sorted([p for p in root.iterdir() if p.is_dir() and (p / "best_params.pt").exists()])
    if not folds and (root / "best_params.pt").exists():
        folds = [root]
    if not folds:
        raise FileNotFoundError(f"No fold checkpoints found under {root}")
    return folds

def _load_skorch_fold(build_model_fn, fold_dir: Path, device="cuda", classes=None):
    f_params = fold_dir / "best_params.pt"
    clf = EEGClassifier(
        module=build_model_fn(),
        device="cuda",
        train_split=None,
        classes=np.array(classes) if classes is not None else None,
    )
    clf.initialize()
    clf.load_params(f_params=f_params)
    return clf

class SkorchEnsemble:
    def __init__(self, build_model_fn, model_root, device="cuda", classes=None):
        self.model_root = Path(model_root)
        self.fold_dirs = _find_fold_dirs(self.model_root)
        self.clfs = [
            _load_skorch_fold(build_model_fn, d, device=device, classes=classes)
            for d in self.fold_dirs
        ]
        print(f"[INFO] Loaded {len(self.clfs)} folds from {self.model_root}")

    def predict_proba(self, dataset_or_X):
        probas = [clf.predict_proba(dataset_or_X) for clf in self.clfs]
        return np.mean(probas, axis=0)

    def predict(self, dataset_or_X):
        return self.predict_proba(dataset_or_X).argmax(axis=1)
