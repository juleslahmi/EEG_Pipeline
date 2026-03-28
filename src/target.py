# src/target_view.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
from torch.utils.data import Dataset

class TargetView(Dataset):
    """
    Wrap a base dataset and switch the target to either 'diagnosis' or 'event'.
    Assumes base dataset exposes:
      - base[i] -> (X, y)  (y is diagnosis by default)
      - base.y            (array-like diagnosis labels)
      - base.event_codes  (array-like event codes, e.g., 201..205), OR
      - base.events       (fallback name)
    Optionally remap raw event codes via event_map to 0..K-1.
    """
    def __init__(self, base: Dataset, target: str = "diagnosis",
                 event_map: Optional[Dict[str, int]] = None):
        self.base = base
        self.target = target
        self.event_map = event_map

        # grab arrays for speed
        self.diag = getattr(base, "y", None)
        self.ev   = getattr(base, "event_codes", None)
        if self.ev is None:
            self.ev = getattr(base, "events", None)

        if self.target == "event" and self.ev is None:
            raise AttributeError("Base dataset must have .event_codes or .events for target='event'.")

        # precompute label array for convenience
        if self.target == "diagnosis":
            self.labels = np.asarray(self.diag, dtype=int)
        else:
            raw = np.asarray(self.ev)
            if self.event_map:
                # map using provided dict (keys may be str or int)
                mapper = {int(k): int(v) for k, v in self.event_map.items()}
                self.labels = np.array([mapper[int(v)] for v in raw], dtype=int)
            else:
                # auto-encode unique raw codes to 0..K-1 (stable order)
                uniq = np.array(sorted(np.unique(raw))).tolist()
                enc  = {int(v): i for i, v in enumerate(uniq)}
                self.labels = np.array([enc[int(v)] for v in raw], dtype=int)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Tuple[Any, int]:
        X, _ = self.base[i]
        return X, int(self.labels[i])

def infer_classes_from(ds: TargetView) -> np.ndarray:
    import numpy as np
    return np.array(sorted(np.unique(ds.labels)), dtype=int)
