import os
from pathlib import Path
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import GroupKFold
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
from braindecode import EEGClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, EpochScoring
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# -----------------------------
# 1) LIST YOUR PATIENTS & LABELS
# -----------------------------
# Example: fill this with your real files + labels (0=control, 1=dyslexic)
PATIENTS = [
    # Each item: (patient_id, path_to_set_or_raw, optional_events_csv, label)
    # If you have epoched EEGLAB .set: set events_csv=None
    ("c06", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c06_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c07", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c07_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c09", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c09_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c10", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c10_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c11", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c11_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c12", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c12_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c13", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c13_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c14", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c14_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c15", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c15_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c16", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c16_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c18", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c18_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c19", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c19_ICA_cleae_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c20", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c20_ICA_clear_blink_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c21", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c21_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c22", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c22_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c23", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c23_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c24", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c24_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),
    ("c25", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Control/c25_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 0),

    ("d05", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d05_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d06", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d06_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d07", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d07_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d08", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d08_ICA_clear_bl_ASR_interpol_reref_elist_be.set", 1),
    ("d10", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d10_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d11", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d11_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d13", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d13_ICA_clear_bl_sac_only_9_ASR_interpol_reref_elist_be.set", 1),
    ("d14", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d14_ICA_clear_bl_sac_11_3_ASR_interpol_reref_elist_be.set", 1),
    ("d15", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d15_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d16", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d16_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d18", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d18_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d19", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d19_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d20", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d20_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    ("d21", "/mnt/c/Users/Mohamed Chetouani/Jules Lahmi/Data/Dyslexic/d21_ICA_clear_bl_sac_ASR_interpol_reref_elist_be.set", 1),
    
    # If you have RAW + events.csv instead:
    # ("subj03", "path/to/subj03_raw.set", "path/to/subj03_events.csv", 1),
]

# Epoch parameters if you need to epoch RAW (ignored for already epoched .set)
TMIN, TMAX = -0.2, 0.8   # seconds
BASELINE = (None, 0)

# Preproc params
L_FREQ, H_FREQ = 4., 38. 
RS_FREQ = 250.0

# -----------------------------
# 2) LOADER -> returns epochs (n_epochs, n_chans, n_times), sfreq, ch_names
# -----------------------------
def load_patient_epochs(file_path):
    file_path = str(file_path)

    epochs = mne.read_epochs_eeglab(file_path)

    # -------- Standard preprocessing on epochs --------
    # (1) Band-pass
    epochs.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    # (2) Resample
    epochs.resample(RS_FREQ, npad="auto")
    # (3) Convert to array in Volts then to µV (values ~ tens)
    X = epochs.get_data() * 1e6  # (n_epochs, n_chans, n_times), microvolts
    # (4) Exponential moving standardization (per channel, over time)
    #     Apply in-place per epoch for stability
    for i in range(X.shape[0]):
        X[i] = exponential_moving_standardize(
            X[i], factor_new=1e-3, init_block_size=1000
        )
    y_dummy = np.zeros(X.shape[0], dtype=np.int64)  # placeholder; labels added later
    return X.astype(np.float32), y_dummy, epochs.info["sfreq"], epochs.ch_names

# -----------------------------
# 3) BUILD A TORCH DATASET ACROSS PATIENTS
# -----------------------------
class PatientEpochsDataset(Dataset):
    def __init__(self):
        self.X = []
        self.y = []
        self.groups = []   # patient id for GroupKFold
        self.patients = [] # parallel list to groups for convenience
        self.n_chans = None
        self.n_times = None

        for pid, fpath, label in PATIENTS:
            X, _, sfreq, chs = load_patient_epochs(fpath)
            if self.n_chans is None:
                self.n_chans = X.shape[1]
                self.n_times = X.shape[2]
            else:
                # sanity: all same shape
                assert X.shape[1] == self.n_chans and X.shape[2] == self.n_times, \
                    f"Shape mismatch: got {X.shape}, expected (n, {self.n_chans}, {self.n_times})"
            # Assign the **patient-level** label to every epoch of that patient
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

# -----------------------------
# 4) GROUPED TRAIN/VALID SPLIT (no patient leakage)
# -----------------------------
full_ds = PatientEpochsDataset()
n_chans, n_times = full_ds.n_chans, full_ds.n_times
groups = np.array(full_ds.groups)

gkf = GroupKFold(n_splits=10)
train_idx, valid_idx = next(gkf.split(np.zeros(len(full_ds)), full_ds.y, groups=groups))

train_set = Subset(full_ds, train_idx)
valid_set = Subset(full_ds, valid_idx)

# -----------------------------
# 5) MODEL + TRAINING (epoch-level)
# -----------------------------
n_classes = 2  # dyslexic vs control
model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    n_times=n_times,
    final_conv_length="auto"
).cuda()

# We are not doing cropped training here; straight CE on epochs:
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

_ = clf.fit(train_set)  # y inferred from dataset

# -----------------------------
# 6) EVALUATION
# -----------------------------
# (a) Epoch-level accuracy
from sklearn.metrics import accuracy_score

def get_labels(ds):
    return np.array([ds[i][1] for i in range(len(ds))], dtype=int)

y_valid = get_labels(valid_set)
y_pred = clf.predict(valid_set)
epoch_acc = accuracy_score(y_valid, y_pred)
print("Epoch-level accuracy:", epoch_acc)

# (b) Patient-level accuracy (aggregate epoch predictions by patient)
def patient_level_accuracy(ds, preds):
    # Map indices back to patient ids
    if isinstance(ds, Subset):
        # ds.indices maps into full_ds
        pids = np.array([full_ds.patients[i] for i in ds.indices])
        labels = np.array([full_ds.y[i] for i in ds.indices], dtype=int)
    else:
        pids = np.array(full_ds.patients)
        labels = np.array(full_ds.y, dtype=int)

    # Majority vote (or mean prob if you prefer predict_proba)
    patient_preds = {}
    patient_labels = {}
    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        votes = preds[idx]
        patient_preds[pid] = int(np.bincount(votes).argmax())
        patient_labels[pid] = int(np.bincount(labels[idx]).argmax())  # they should all match

    pp = np.array([patient_preds[k] for k in patient_preds.keys()], dtype=int)
    yy = np.array([patient_labels[k] for k in patient_labels.keys()], dtype=int)
    return accuracy_score(yy, pp), patient_preds, patient_labels

pat_acc, pat_pred, pat_true = patient_level_accuracy(valid_set, y_pred)
print("Patient-level accuracy:", pat_acc)
