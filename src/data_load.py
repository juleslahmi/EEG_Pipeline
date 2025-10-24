from dis import disco
import os
import mne
import numpy as np
from pathlib import Path
from braindecode.preprocessing import exponential_moving_standardize
from torch.utils.data import Dataset, Subset
import torch

TMIN, TMAX = -0.2, 0.8
BASELINE = (None, 0)

# Preproc params
L_FREQ, H_FREQ = 4., 38. 
RS_FREQ = 250.0

def load_patient_epochs(file_path):
    file_path = str(file_path)

    epochs = mne.read_epochs_eeglab(file_path)

    #Band-pass
    epochs.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    #Resample
    epochs.resample(RS_FREQ, npad="auto")
    #Convert in Volts
    X = epochs.get_data() * 1e6  # (n_epochs, n_chans, n_times), microvolts
    #Exponential moving standardization (per channel, over time)
    for i in range(X.shape[0]):
        X[i] = exponential_moving_standardize(
            X[i], factor_new=1e-3, init_block_size=1000
        )
    y_dummy = np.zeros(X.shape[0], dtype=np.int64)
    return X.astype(np.float32), y_dummy, epochs.info["sfreq"], epochs.ch_names

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
    def __init__(self, patients):
        self.X = []
        self.y = []
        self.groups = []  
        self.patients = []
        self.n_chans = None
        self.n_times = None


        for pid, fpath, label in patients:
            X, _, sfreq, chs = load_patient_epochs(fpath)
            if self.n_chans is None:
                self.n_chans = X.shape[1]
                self.n_times = X.shape[2]
            else:
                assert X.shape[1] == self.n_chans and X.shape[2] == self.n_times, \
                    f"Shape mismatch: got {X.shape}, expected (n, {self.n_chans}, {self.n_times})"

            y = np.full(X.shape[0], int(label), dtype=np.int64)

            self.X.append(X)
            self.y.append(y)
            self.groups.extend([pid] * X.shape[0])
            self.patients.extend([pid] * X.shape[0])

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return (X_epoch, y_label)
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


def load_dataset(data_path, extension):
    patients = discover_subjects(data_path, extension)
    dataset = PatientEpochsDataset(patients)
    return {
        "patients": patients,
        "dataset": dataset,
        "n_chans": dataset.n_chans,
        "n_times": dataset.n_times,
        "n_subjects": len(set(dataset.groups))
    }

def save_dataset_cache(bundle: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = bundle["dataset"]
    np.savez_compressed(
        path,
        X=ds.X, y=ds.y,
        groups=np.array(ds.groups, dtype=object),
        patients=np.array(ds.patients, dtype=object),
        n_chans=ds.n_chans, n_times=ds.n_times
    )

def load_dataset_from_cache(path: str | Path):
    d = np.load(path, allow_pickle=True)
    # lightweight Dataset wrapper that reads from arrays
    import torch
    from torch.utils.data import Dataset

    class ArrayDataset(Dataset):
        def __init__(self, X, y, groups, patients, n_chans, n_times):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64)
            self.groups = list(groups)
            self.patients = list(patients)
            self.n_chans = int(n_chans)
            self.n_times = int(n_times)

        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i): 
            return torch.from_numpy(self.X[i]), int(self.y[i])

    ds = ArrayDataset(
        d["X"], d["y"], d["groups"], d["patients"], d["n_chans"], d["n_times"]
    )
    return {
        "patients": list(d["patients"]),
        "dataset": ds,
        "n_chans": int(d["n_chans"]),
        "n_times": int(d["n_times"]),
        "n_subjects": len(set(ds.groups)),
    }
