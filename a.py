from braindecode.datasets import MOABBDataset
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events

# Bandpass 4–38 Hz, standardize
preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    Preprocessor('filter', l_freq=4., h_freq=38.),
    Preprocessor('resample', sfreq=250)
]
preprocess(dataset, preprocessors)

# Cut into trials (windows)
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0,
    trial_stop_offset_samples=0, preload=True
)

from braindecode.models import ShallowFBCSPNet
import torch

n_classes = len(set(windows_dataset.datasets[0].y))
n_chans = windows_dataset[0][0].shape[0]
input_time_length = windows_dataset[0][0].shape[1]
n_times = windows_dataset[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans, n_classes, n_times=n_times,
    final_conv_length="auto"
)
model = model.cuda()

from braindecode import EEGClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

train_set = windows_dataset.split('train')['train']
valid_set = windows_dataset.split('train')['valid']

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=Adam,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.01,
    batch_size=64,
    max_epochs=10,
    callbacks=[
        LRScheduler(policy=CosineAnnealingLR, T_max=10)
    ],
    device='cuda'
)
clf.fit(train_set)

score = clf.score(valid_set)
print("Validation accuracy:", score)

