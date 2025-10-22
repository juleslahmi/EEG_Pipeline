from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import numpy as np


def make_splits(dataset, scheme="GroupKFold", n_splits=10, seed=42):
    y = np.array(dataset.y)
    groups = np.array(dataset.groups)

    if scheme.lower() in ["loso", "leaveoneout", "logo"]:
        splitter = LeaveOneGroupOut()
    else:
        splitter = GroupKFold(n_splits=n_splits)

    splits = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        valid_subjects = np.unique(np.array(groups)[valid_idx]).tolist()
        splits.append({
            "fold": fold,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "valid_subjects": valid_subjects
        })

    return splits
