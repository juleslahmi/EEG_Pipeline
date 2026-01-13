from dis import disco
import os
import mne
import numpy as np
from pathlib import Path
import json
import hashlib
from braindecode.preprocessing import exponential_moving_standardize
from torch.utils.data import Dataset, Subset
import torch
import re

BASELINE = (None, 0)

# Preproc params
L_FREQ, H_FREQ = 4., 38. 
import re

def load_patient_epochs(file_path, tmin=0.1, tmax=0.2, freq=40, drop_channels=None):
    epochs = mne.read_epochs_eeglab(str(file_path), verbose=False)

    # --- drop common EOG channels if present ---
    eog_keywords = ('EOG', 'VEOG', 'HEOG', 'LOC', 'ROC', 'M1', 'M2')   # add names you use
    to_drop = [ch for ch in epochs.ch_names if any(k in ch.upper() for k in eog_keywords)]
    
    if drop_channels:
        to_drop.extend([ch for ch in epochs.ch_names if ch in drop_channels])
    print(to_drop)
    if to_drop:
        print(f"Keeping channels: {[ch for ch in epochs.ch_names if ch not in to_drop]}")
        epochs.drop_channels(to_drop)
    # ------------------------------------------

    # id -> label string
    id_to_label = {v: k for k, v in epochs.event_id.items()}
    raw_ids = epochs.events[:, -1]

    # extract 2011, 2021, ... from the label
    stim_codes = []
    for ev_id in raw_ids:
        label = id_to_label.get(int(ev_id), str(ev_id))
        m = re.search(r"\((\d+)\)", label)
        if not m:
            raise ValueError(f"Could not find code in label '{label}'")
        stim_codes.append(int(m.group(1)))
    stim_codes = np.array(stim_codes, dtype=np.int64)

    # then your existing filtering/resampling/standardization
    epochs.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    epochs.crop(tmin=tmin, tmax=tmax)
    epochs.resample(freq, npad="auto")
    X = epochs.get_data() * 1e6
    for i in range(X.shape[0]):
        X[i] = exponential_moving_standardize(X[i], factor_new=1e-3, init_block_size=1000)

    return X.astype(np.float32), stim_codes, epochs.info["sfreq"], epochs.ch_names



def discover_subjects(data_root: str | Path, ext=".set") -> list[tuple[str, str, int]]: 
    patients = []
    data_root = Path(data_root)
    class_dirs = {"Control" : 0, "Dyslexic" : 1}
    for class_name in class_dirs.keys() :
        label = class_dirs[class_name]
        class_path = data_root/class_name
        if not class_path.exists():
            print(f"class path {class_path} does not exist")
            continue
        
        for file_path in class_path.rglob(f"*{ext}"):
            stem = file_path.stem
            patient_id = stem.split("_")[0]
            patients.append((patient_id, str(file_path), label))

    return patients

class PatientEpochsDataset(Dataset):
    def __init__(self, patients, target='diagnosis', event_map=None, tmin=0.1, tmax=0.2, freq=40, drop_channels=None):
        self.X = []
        self.y = []
        self.groups = []  
        self.patients = []
        self._events_chunks = []  # accumulate arrays per file, then concat once
        self.events = None
        self.n_chans = None
        self.n_times = None
        self.target = target
        self.event_map = {int(k): int(v) for k, v in event_map.items()} if event_map else None
        self.freq = freq
        self.drop_channels = drop_channels


        for pid, fpath, label in patients:
            X, events, sfreq, chs = load_patient_epochs(fpath, tmin=tmin, tmax=tmax, freq=self.freq, drop_channels=self.drop_channels)
            if self.n_chans is None:
                self.n_chans = X.shape[1]
                self.n_times = X.shape[2]
            else:
                assert X.shape[1] == self.n_chans and X.shape[2] == self.n_times, \
                    f"Shape mismatch: got {X.shape}, expected (n, {self.n_chans}, {self.n_times})"
                
            if self.target == "diagnosis":
                y = np.full(X.shape[0], int(label), dtype=np.int64)
            elif self.target == "event":
                if self.event_map:
                    y = np.array([self.event_map.get(str(e), e) for e in events], dtype=np.int64)
                else:
                    y = events.astype(np.int64)
            else:
                raise ValueError("Target must be either 'diagnosis' or 'event'.")

            self.X.append(X)
            self.y.append(y)
            self.groups.extend([pid] * X.shape[0])
            self.patients.extend([pid] * X.shape[0])
            self._events_chunks.append(events.astype(np.int64))

        self.events = np.concatenate(self._events_chunks, axis=0) if self._events_chunks else np.array([], dtype=np.int64)
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return (X_epoch, y_label)
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


def load_dataset(data_path, extension, target: str = "diagnosis", event_map: dict | None = None, tmin=0.1, tmax=0.2, freq: int = 40, drop_channels=None):
    patients = discover_subjects(data_path, extension)
    dataset = PatientEpochsDataset(patients, target=target, event_map=event_map, tmin=tmin, tmax=tmax, freq=freq, drop_channels=drop_channels)
    return {
        "patients": patients,
        "dataset": dataset,
        "n_chans": dataset.n_chans,
        "n_times": dataset.n_times,
        "n_subjects": len(set(dataset.groups)),
        "target": target,
        "freq": freq,
    }

def save_dataset_cache(bundle: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = bundle["dataset"]
    target = bundle.get("target", "diagnosis")
    event_map = bundle.get("event_map", None)

    event_map_sha = ""
    if event_map:
        event_map_sha = hashlib.sha1(
            json.dumps({str(k): int(v) for k, v in event_map.items()}, sort_keys=True).encode()
        ).hexdigest()[:8]
    
    np.savez_compressed(
        path,
        X=ds.X, y=ds.y,
        groups=np.array(ds.groups, dtype=object),
        patients=np.array(ds.patients, dtype=object),
        n_chans=ds.n_chans, n_times=ds.n_times,
        target=np.array([target], dtype=object),
        events=ds.events.astype(np.int64),
        event_map=np.array([json.dumps(event_map) if event_map else ""], dtype=object),
        event_map_sha=np.array([event_map_sha], dtype=object),
    )

def load_dataset_from_cache(path: str | Path):
    d = np.load(path, allow_pickle=True)
    # lightweight Dataset wrapper that reads from arrays
    import torch
    from torch.utils.data import Dataset

    class ArrayDataset(Dataset):
        def __init__(self, X, y, groups, patients, n_chans, n_times, events):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64)
            self.groups = list(groups)
            self.patients = list(patients)
            self.n_chans = int(n_chans)
            self.n_times = int(n_times)
            self.events = events.astype(np.int64)

        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i): 
            return torch.from_numpy(self.X[i]), int(self.y[i])

    ds = ArrayDataset(
        d["X"], d["y"], d["groups"], d["patients"], d["n_chans"], d["n_times"], d["events"]
    )
    return {
        "patients": list(d["patients"]),
        "dataset": ds,
        "n_chans": int(d["n_chans"]),
        "n_times": int(d["n_times"]),
        "n_subjects": len(set(ds.groups)),
        "target": str(d["target"][0]) if "target" in d else "diagnosis",
        "event_map": (json.loads(str(d["event_map"][0])) if "event_map" in d and str(d["event_map"][0]) else None),
        "event_map_sha": str(d["event_map_sha"][0]) if "event_map_sha" in d else "",
    }
