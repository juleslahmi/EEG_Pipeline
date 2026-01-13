# src/channel_groups.py
"""
Channel groupings based on anatomical regions for EEG ablation experiments.
All channels available in dataset:
Fp1 AF7 AF3 F1 F3 F5 F7 FT7 FC5 FC3 FC1 C1 C3 C5 T7 TP7 CP5 CP3 CP1 P1 P3 P5 P7 P9 
PO7 PO3 O1 Iz Oz POz Pz CPz Fpz Fp2 AF8 AF4 AFz Fz F2 F4 F6 F8 FT8 FC6 FC4 FC2 FCz 
Cz C2 C4 C6 T8 TP8 CP6 CP4 CP2 P2 P4 P6 P8 P10 PO8 PO4 O2 M1 M2 HEOG1 HEOG2 VEOG1 VEOG2
"""

CHANNEL_GROUPS = {
    "frontal": [
        "Fp1", "Fp2", "Fpz",
        "AF7", "AF8", "AF3", "AF4", "AFz",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz"
    ],
    "frontocentral": [
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
        "FT7", "FT8"
    ],
    "central": [
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz"
    ],
    "centroparietal": [
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz"
    ],
    "temporal": [
        "T7", "T8",
        "TP7", "TP8"
    ],
    "parietal": [
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "Pz"
    ],
    "parietooccipital": [
        "PO3", "PO4", "PO7", "PO8", "POz"
    ],
    "occipital": [
        "O1", "O2", "Oz", "Iz"
    ],
    "FRONTAL": [
        "Fp1", "Fp2", "Fpz",
        "AF7", "AF8", "AF3", "AF4", "AFz",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz",
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
        "FT7", "FT8"
    ],
    "CENTRAL": [
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz", 
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz"
    ],
    "TEMPORAL": [
        "T7", "T8",
        "TP7", "TP8"
    ],
    "PARIETAL": [
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "Pz",
        "PO3", "PO4", "PO7", "PO8", "POz",
        "O1", "O2", "Oz", "Iz"
    ],
}

# All EEG channels (excluding EOG and mastoid)
ALL_EEG_CHANNELS = (
    CHANNEL_GROUPS["frontal"] + 
    CHANNEL_GROUPS["frontocentral"] + 
    CHANNEL_GROUPS["central"] + 
    CHANNEL_GROUPS["centroparietal"] + 
    CHANNEL_GROUPS["temporal"] + 
    CHANNEL_GROUPS["parietal"] + 
    CHANNEL_GROUPS["parietooccipital"] + 
    CHANNEL_GROUPS["occipital"]
)

def get_channels_to_drop(drop_region):
    """Return list of channels to drop for a given region ablation."""
    return CHANNEL_GROUPS.get(drop_region, [])

def get_channels_to_keep(drop_region):
    """Return list of channels to keep (all EEG except drop_region)."""
    drop_set = set(get_channels_to_drop(drop_region))
    return [ch for ch in ALL_EEG_CHANNELS if ch not in drop_set]