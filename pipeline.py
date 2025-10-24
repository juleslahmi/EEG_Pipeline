from braindecode.datasets import MOABBDataset

from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
import torch

from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from numpy import multiply

dataset = MOABBDataset(dataset_name="BNCI2014_001")

factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    Preprocessor(lambda data:multiply(data,factor)),
    Preprocessor('filter', l_freq=4., h_freq=38.),
    Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000)
]
preprocess(dataset, preprocessors, n_jobs=8)

torch.backends.cudnn.benchmark = True
cuda = torch.cuda.is_available()
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
n_chans = dataset[0][0].shape[0]
input_time_length = dataset[0][0].shape[1]
n_times=1125

model = ShallowFBCSPNet(
    n_chans, n_classes, n_times=n_times,
    final_conv_length="auto"
)
model = model.cuda()
model.to_dense_prediction_model()




trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

n_preds_per_input = model.get_output_shape()[2]

windows_dataset = create_windows_from_events(
    dataset, 
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
    window_size_samples=n_times,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False
)








splits = windows_dataset.split('session')
train_set = splits['0train']
valid_set = splits['1test']


lr = 0.0625 * 0.01
weight_decay = 0

batch_size = 64

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.cross_entropy,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    batch_size=batch_size,
    optimizer__weight_decay=weight_decay,
    max_epochs=20,
    callbacks=[
        LRScheduler(policy=CosineAnnealingLR, T_max=10)
    ],
    device='cuda',
    classes=classes 
)
_ = clf.fit(train_set, y=None, epochs=20)

try:
    y_valid = valid_set.get_metadata()['target'].to_numpy()
except AttributeError:
    # e.g., if it's a torch.utils.data.Subset
    y_valid = np.array([valid_set[i][1] for i in range(len(valid_set))])

score = clf.score(valid_set, y_valid)
print("Validation accuracy:", score)

